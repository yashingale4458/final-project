"""One question, one language model call, answered from Merrowdale's documents and from
nothing else. The assistant classifies what it was asked, answers the kinds it may
answer, and refuses the rest. It knows nothing of any single application or substation:
the service answers those from the records."""

import re

from agentic.agents import complete, parse_labelled_response
from chat.record_answer import canonical_reference
from config import config
from knowledge.standard_reading import read_connection_standard
from knowledge.vector_index import search_documents
from tools.retrieval_tools import read_documentation_to_provide, read_reviewer_checks

CHAT_LABELS = ["QUESTION TYPE", "APPLICATION", "ANSWER"]
_reference_material = {}


def _material(collection, document_entries):
    """Every standard's requirements and documentation, and every guide's checks, read once."""
    key = id(collection)
    if key not in _reference_material:
        standards, guides = [], []
        for technology in config.CONNECTION_STANDARD_DOCUMENTS:
            standard = read_connection_standard(technology, document_entries)
            requirements = "\n".join(f"  {index}. {requirement['requirement_name']}: "
                                     f"{requirement['requirement_sentence']}"
                                     for index, requirement in enumerate(standard["requirements"], 1))
            standards.append({
                "technology": technology,
                "document_name": standard["document_name"],
                "requirements": standard["requirements"],
                "text": (f"{config.TECHNOLOGY_NAMES[technology]} ({technology}), from "
                         f"{standard['document_name']}:\n{requirements}\n  Documentation to provide: "
                         f"{read_documentation_to_provide(collection, technology)}"),
            })
        for sentence_type, document_name in config.SENTENCE_TYPE_GUIDE_DOCUMENTS.items():
            guides.append({"sentence_type": sentence_type, "document_name": document_name,
                           "checks": read_reviewer_checks(collection, sentence_type)})
        _reference_material[key] = {"standards": standards, "guides": guides}
    return _reference_material[key]


def _strip_edges(text):
    return re.sub(config.RESPONSE_DECORATION_EDGE_PATTERN, "", text.strip())


def _named_in(reference, texts):
    """Whether the question or the history names the reference."""
    lowered = [text.lower() for text in texts]
    if re.fullmatch(config.CANONICAL_APPLICATION_REFERENCE_PATTERN, reference):
        number = str(int(reference[len(config.APPLICATION_REFERENCE_PREFIX):]))
        pattern = config.NUMBER_IN_TEXT_PATTERN.format(number=number)
        return any(reference.lower() in text or re.search(pattern, text) for text in lowered)
    return any(reference.lower() in text for text in lowered)


def _technologies_in(text):
    lowered = text.lower()
    return [technology for technology, pattern in config.CHAT_TECHNOLOGY_PATTERNS.items()
            if re.search(pattern, lowered)]


def _substations_in(text):
    found = []
    for match in re.findall(config.SUBSTATION_IN_TEXT_PATTERN, text):
        if match.upper() not in found:
            found.append(match.upper())
    return found


# --------------------------------------------------------------------------- offline classifier
_PREDICTION_WORDS = r"\b(how long|when will|network work|predict|delay|working days|need work)\b"
_GUIDE_WORDS = r"\b(statement|statements|reviewer|sentence|guide|evidence)\b"
_STANDARD_WORDS = (r"\b(require|requires|requirement|requirements|standard|documentation|provide|"
                   r"rated|power factor|register)\b")
_RULE_WORDS = (r"\b(trigger|triggers|looped|cut-out|cut out|phase|phases|kva|kw|convert|revision|"
               r"approve|approval|refer|return|blank|advisor|customer service|premises|queue|"
               r"heating failed|notify|fast track|decide|decision|policy|rules?)\b")


def _offline_reply(question_text, conversation, passages, material):
    question = question_text.lower()
    references = re.findall(config.APPLICATION_REFERENCE_IN_TEXT_PATTERN, question_text)
    application = references[0] if references else "none"
    technologies = _technologies_in(question_text)
    if re.search(_PREDICTION_WORDS, question) and (references or "application" in question):
        kind = config.CHAT_TYPE_OUTCOME_PREDICTION
    elif references:
        kind = config.CHAT_TYPE_APPLICATION_STATUS
    elif _substations_in(question_text) or ("how many" in question and "substation" in
                                            (question + conversation["previous_question"].lower())):
        kind = config.CHAT_TYPE_SUBSTATION_FIGURE
    elif re.search(_GUIDE_WORDS, question):
        kind = config.CHAT_TYPE_STATEMENT_GUIDANCE
    elif technologies and re.search(_STANDARD_WORDS, question):
        kind = config.CHAT_TYPE_CONNECTION_STANDARD
    elif re.search(_RULE_WORDS, question) or (technologies and passages):
        kind = config.CHAT_TYPE_CONNECTION_RULES
    else:
        kind = config.CHAT_TYPE_OUT_OF_SCOPE

    answer = config.CHAT_NO_VALUE_REPLY
    if kind == config.CHAT_TYPE_CONNECTION_STANDARD:
        chosen = [standard for standard in material["standards"] if standard["technology"] in technologies]
        answer = " ".join(
            f"The {config.TECHNOLOGY_NAMES[standard['technology']]} standard "
            f"({standard['document_name']}) sets these requirements: "
            + " ".join(f"({index}) {requirement['requirement_name']}: {requirement['requirement_sentence']}"
                       for index, requirement in enumerate(standard["requirements"], start=1))
            for standard in chosen)
    elif kind == config.CHAT_TYPE_STATEMENT_GUIDANCE and passages:
        guide = next((guide for guide in material["guides"]
                      if guide["document_name"] == passages[0]["document_name"]), None)
        answer = (f"{guide['document_name']} says the reviewer checks this: {guide['checks']}"
                  if guide else f"{passages[0]['document_name']} says: {passages[0]['chunk_text']}")
    elif kind == config.CHAT_TYPE_CONNECTION_RULES and passages:
        top = passages[0]
        answer = f"{top['document_name']}, section {top['section_number']}, says: {top['chunk_text']}"
    return (f"QUESTION TYPE: {kind}\nAPPLICATION: {application}\n"
            f"ANSWER: {answer if answer.strip() else config.CHAT_NO_VALUE_REPLY}")


# --------------------------------------------------------------------------- the assistant
def answer_from_documents(collection, document_entries, question_text, conversation):
    """Classify the question and answer it from the documents in one call, or refuse."""
    agent = config.AGENT_NAME_CONNECTION_ASSISTANT
    found = search_documents(collection, [question_text, conversation["search_text"]])
    passages, seen = [], set()
    for passage in sorted([passage for result in found for passage in result],
                          key=lambda passage: passage["distance"]):
        if passage["chunk_id"] not in seen:
            seen.add(passage["chunk_id"])
            passages.append(passage)
    material = _material(collection, document_entries)

    passage_text = "\n\n".join(f"[{passage['document_name']}, section {passage['section_number']}] "
                               f"{passage['chunk_text']}" for passage in passages)
    standards_text = "\n\n".join(standard["text"] for standard in material["standards"])
    guides_text = "\n".join(f"- {guide['sentence_type']} ({guide['document_name']}): {guide['checks']}"
                            for guide in material["guides"])
    kinds = "\n".join(f"- {kind}" for kind in config.CHAT_TYPE_OPTIONS)
    prompt = (
        f"You are the connection assistant of {config.ORGANISATION_NAME}. You answer questions "
        f"about home low-carbon connections from the documents below and from nothing else. You "
        f"know nothing about any individual application or substation: never guess one.\n"
        f"\nTHE CONVERSATION SO FAR\n{conversation['history_text']}\n"
        f"\nTHE QUESTION\n{question_text}\n"
        f"\nPASSAGES FOUND FOR IT\n{passage_text}\n"
        f"\nEVERY TECHNOLOGY'S REQUIREMENTS, EACH FROM ITS OWN STANDARD\n{standards_text}\n"
        f"\nWHAT EACH STATEMENT GUIDE'S REVIEWER CHECKS\n{guides_text}\n"
        f"\nClassify the question as exactly one of these kinds:\n{kinds}\n"
        f"Questions about one application's status are {config.CHAT_TYPE_APPLICATION_STATUS}; "
        f"questions asking whether one will need network work or how long it will take are "
        f"{config.CHAT_TYPE_OUTCOME_PREDICTION}; questions about how many of a technology are at "
        f"a substation are {config.CHAT_TYPE_SUBSTATION_FIGURE}. Answer only "
        f"{', '.join(config.CHAT_ANSWERED_TYPES)} questions, holding each requirement to its own "
        f"technology's standard. Write {config.CHAT_NO_VALUE_REPLY} in any field that does not apply.\n"
        f"\nReply in plain text with exactly these fields, each on a new line:\n"
        f"QUESTION TYPE: one kind from the list, on one line\n"
        f"APPLICATION: the application reference the question is about, or {config.CHAT_NO_VALUE_REPLY}\n"
        f"ANSWER: the answer from the documents, or {config.CHAT_NO_VALUE_REPLY}\n")

    reply = complete(prompt, agent, lambda: _offline_reply(question_text, conversation, passages, material))
    fields = parse_labelled_response(reply["response_text"], CHAT_LABELS, agent)
    kind_text, application_text = fields["QUESTION TYPE"], fields["APPLICATION"]
    question_type = _strip_edges(kind_text).upper()
    application_field = _strip_edges(application_text)
    if "\n" in kind_text or "\n" in application_text or question_type not in config.CHAT_TYPE_OPTIONS:
        raise ValueError(f"{agent} gave a question type or application that cannot be read")
    if not application_field:
        raise ValueError(f"{agent} left the application empty")

    if question_type in config.CHAT_ANSWERED_TYPES:
        if re.search(config.APPLICATION_REFERENCE_IN_TEXT_PATTERN, question_text):
            question_type = config.CHAT_TYPE_APPLICATION_STATUS
        elif re.search(config.SUBSTATION_IN_TEXT_PATTERN, question_text):
            question_type = config.CHAT_TYPE_SUBSTATION_FIGURE
    answer = fields["ANSWER"]
    if question_type in config.CHAT_ANSWERED_TYPES and \
            _strip_edges(answer).lower() == config.CHAT_NO_VALUE_REPLY:
        raise ValueError(f"{agent} classified an answerable question but gave no answer")

    application_reference = None
    if question_type in config.CHAT_APPLICATION_TYPES:
        named = re.findall(config.APPLICATION_REFERENCE_IN_TEXT_PATTERN, question_text)
        if named:
            application_reference = named[0]
        elif application_field.lower() != config.CHAT_NO_VALUE_REPLY:
            application_reference = application_field
        if application_reference is not None:
            application_reference = canonical_reference(application_reference)
            if not _named_in(application_reference, [question_text, conversation["history_text"]]):
                raise ValueError(f"{agent} named an application the conversation never named")

    substation_reference = technology = None
    if question_type == config.CHAT_TYPE_SUBSTATION_FIGURE:
        substations = _substations_in(question_text) or _substations_in(conversation["previous_question"])
        technologies = _technologies_in(question_text) or _technologies_in(conversation["previous_question"])
        if len(substations) == 1 and len(technologies) == 1:
            substation_reference, technology = substations[0], technologies[0]

    closest = passages[0]["distance"] if passages else 1.0
    answered = question_type in config.CHAT_ANSWERED_TYPES and closest <= config.CHAT_MAX_RELEVANCE_DISTANCE
    was_refused = False
    if not answered and question_type not in config.CHAT_RECORD_TYPES:
        question_type, answer, was_refused = (config.CHAT_TYPE_OUT_OF_SCOPE,
                                              config.CHAT_REFUSAL_MESSAGE, True)
        application_reference = substation_reference = technology = None

    sources = []
    if answered:
        if question_type == config.CHAT_TYPE_CONNECTION_STANDARD:
            sources += [standard["document_name"] for standard in material["standards"]]
        if question_type == config.CHAT_TYPE_STATEMENT_GUIDANCE:
            sources += [guide["document_name"] for guide in material["guides"]]
        sources += [passage["document_name"] for passage in passages]
    return {
        "question_type": question_type,
        "application_reference": application_reference,
        "substation_reference": substation_reference,
        "technology": technology,
        "answer": answer,
        "source_documents": list(dict.fromkeys(sources)),
        "closest_distance": float(closest),
        "was_refused": was_refused,
    }
