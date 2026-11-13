import pytest


@pytest.fixture(scope="session")
def processed_frames():
    import pandas as pd

    from config import config

    loaded = {"problem": None}
    try:
        for file_path in [config.CLEAN_APPLICATION_HISTORY_TRAIN_FILE,
                          config.CLEAN_APPLICATION_HISTORY_EVAL_FILE,
                          config.CLEAN_CONNECTION_LOGS_TRAIN_FILE,
                          config.CLEAN_CONNECTION_LOGS_EVAL_FILE,
                          config.CLEAN_STATEMENT_SENTENCES_FILE]:
            loaded[file_path] = pd.read_csv(file_path, keep_default_na=False, na_values=[""],
                                            low_memory=False)
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The cleaned extracts could not be read ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Every check on cleaning, features and models reads the files the data "
            "pipeline writes, so the pipeline has to have run and written all five."
        )
    return loaded


@pytest.fixture(scope="session")
def raw_frames():
    import pandas as pd

    from config import config

    loaded = {"problem": None}
    try:
        for file_path in [config.RAW_APPLICATION_HISTORY_TRAIN_FILE,
                          config.RAW_APPLICATION_HISTORY_EVAL_FILE,
                          config.RAW_CONNECTION_LOGS_TRAIN_FILE,
                          config.RAW_CONNECTION_LOGS_EVAL_FILE,
                          config.RAW_STATEMENT_SENTENCES_FILE]:
            loaded[file_path] = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The raw extracts supplied with the project could not be read ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). The data folder has to stay as it was supplied."
        )
    return loaded


@pytest.fixture(scope="session")
def saved_models():
    from config import config

    loaded = {"problem": None}
    try:
        from modeling.model_registry import load_feature_schema, load_model

        loaded.update({
            "network_work": load_model(config.NETWORK_WORK_MODEL_FILE),
            "duration": load_model(config.DURATION_MODEL_FILE),
            "sentence": load_model(config.SENTENCE_MODEL_FILE),
            "network_work_schema": list(load_feature_schema(config.NETWORK_WORK_SCHEMA_FILE)),
            "duration_schema": list(load_feature_schema(config.DURATION_SCHEMA_FILE)),
        })
        for name, label, method_name in [("network_work", "network work", "predict_proba"),
                                         ("duration", "duration", "predict"),
                                         ("sentence", "sentence", "predict")]:
            assert callable(getattr(loaded[name], method_name, None)), (
                "load_model returned "
                + type(loaded[name]).__name__
                + " for the "
                + label
                + " model, where it returns the fitted model training saved, which has "
                + method_name
                + "."
            )
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The saved models and their feature schemas could not be loaded ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Model training saves them, and every check on the models scores what "
            "training saved."
        )
    return loaded


@pytest.fixture(scope="session")
def live_files():
    from config import config

    loaded = {"problem": None}
    try:
        from data_pipeline.dataset_loader import load_raw_dataset
        from tools.application_data_tools import (fetch_application, load_application_documents,
                                                  load_open_applications)

        loaded.update({
            "raw_applications": load_raw_dataset(config.OPEN_APPLICATIONS_FILE),
            "applications": load_open_applications(),
            "documents": load_application_documents(),
        })
        for name, column_name in [("raw_applications", config.APPLICATION_REFERENCE_COLUMN),
                                  ("applications", config.TECHNOLOGY_COLUMN),
                                  ("documents", config.DOCUMENT_TEXT_COLUMN)]:
            assert column_name in getattr(loaded[name], "columns", []), (
                "The live "
                + name.replace("_", " ")
                + " came back as "
                + type(loaded[name]).__name__
                + " where they are a table holding the column "
                + column_name
                + "."
            )
        first_reference = loaded["applications"][config.APPLICATION_REFERENCE_COLUMN].iloc[0]
        first_application = fetch_application(loaded["applications"], first_reference)
        assert isinstance(first_application, dict) and first_application.get(
            config.APPLICATION_REFERENCE_COLUMN) == first_reference, (
            "fetch_application returned "
            + str(first_application)[:200]
            + " for "
            + str(first_reference)
            + ", where it returns that application's answers keyed by their columns."
        )
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The live files could not be loaded ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). The queue, the screening of a live application and the workflow all start "
            "from the open applications and the statements attached to them, read through the "
            "same cleaning as the extracts."
        )
    return loaded


@pytest.fixture(scope="session")
def knowledge_collection():
    from config import config

    opened = {"problem": None}
    try:
        from knowledge.vector_index import open_vector_index

        opened["collection"] = open_vector_index(config.VECTOR_STORE_DIRECTORY)
        assert callable(getattr(opened["collection"], "query", None)), (
            "open_vector_index returned "
            + type(opened["collection"]).__name__
            + " where it returns the collection the passages are stored in."
        )
    except (Exception, SystemExit) as error:
        opened["problem"] = (
            "The knowledge index could not be opened ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Knowledge indexing builds it, and every check on retrieval reads it."
        )
    return opened


@pytest.fixture(scope="session")
def document_texts():
    from config import config

    read = {"problem": None}
    try:
        from knowledge.pdf_extraction import extract_text_from_directory

        read["documents"] = extract_text_from_directory(config.DOCS_DIRECTORY)
        assert isinstance(read["documents"], list) and len(read["documents"]) > 0, (
            "extract_text_from_directory returned "
            + type(read["documents"]).__name__
            + " where it returns one entry for each PDF in the folder."
        )
        unkeyed = [entry for entry in read["documents"]
                   if not isinstance(entry, dict)
                   or not {"document_name", "document_text"} <= set(entry)]
        assert len(unkeyed) == 0, (
            str(len(unkeyed))
            + " entries read from the document folder do not name their document and carry its "
            "text, for example "
            + str(unkeyed[:1])[:200]
            + "."
        )
    except (Exception, SystemExit) as error:
        read["problem"] = (
            "The document folder could not be read ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Every check on the knowledge base starts from the text of the sixteen "
            "documents."
        )
    return read


@pytest.fixture(scope="session")
def document_passages(document_texts):
    cut = {"problem": document_texts["problem"]}
    if cut["problem"] is not None:
        return cut
    try:
        from knowledge.chunk_splitting import build_chunk_records, split_documents_into_sections

        cut["documents"] = document_texts["documents"]
        cut["sections"] = split_documents_into_sections(cut["documents"])
        cut["chunks"] = build_chunk_records(cut["sections"])
        assert isinstance(cut["sections"], list) and isinstance(cut["chunks"], list), (
            "Splitting the documents returned "
            + type(cut["sections"]).__name__
            + " and cutting the sections "
            + type(cut["chunks"]).__name__
            + ", where each returns a list of entries."
        )
        for entry_name, entries, keys in [
                ("section", cut["sections"], ["document_name", "section_number", "section_text"]),
                ("passage", cut["chunks"], ["chunk_id", "chunk_text", "document_name"])]:
            unkeyed = [entry for entry in entries
                       if not isinstance(entry, dict) or not set(keys) <= set(entry)]
            assert len(entries) > 0 and len(unkeyed) == 0, (
                "Every "
                + entry_name
                + " carries "
                + str(keys)
                + ", but "
                + str(len(unkeyed))
                + " of the "
                + str(len(entries))
                + " do not, for example "
                + str(unkeyed[:1])[:200]
                + "."
            )
    except (Exception, SystemExit) as error:
        cut["problem"] = (
            "The documents could not be split into sections and cut into passages ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Every check on the knowledge base starts from the passages the sixteen "
            "documents are cut into."
        )
    return cut


@pytest.fixture(scope="session")
def connection_standards(document_texts):
    from config import config

    read = {"problem": document_texts["problem"], "standards": {}}
    if read["problem"] is not None:
        return read
    try:
        from knowledge.standard_reading import read_connection_standard

        for technology in config.CONNECTION_STANDARD_DOCUMENTS:
            read["standards"][technology] = read_connection_standard(
                technology, document_texts["documents"])
            assert isinstance(read["standards"][technology], dict) and isinstance(
                read["standards"][technology].get("requirements"), list), (
                "read_connection_standard returned "
                + str(read["standards"][technology])[:200]
                + " for "
                + technology
                + ", where it returns the standard with its requirements."
            )
            unkeyed = [requirement for requirement
                       in read["standards"][technology]["requirements"]
                       if not isinstance(requirement, dict)
                       or not {"requirement_kind", "requirement_name", "requirement_sentence",
                               "rules"} <= set(requirement)]
            assert len(unkeyed) == 0, (
                "The standard read for "
                + technology
                + " holds requirements without their kind, name, sentence and rules, for "
                "example "
                + str(unkeyed[:1])[:200]
                + "."
            )
    except (Exception, SystemExit) as error:
        read["problem"] = (
            "The connection standards could not be read from their documents ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Each application is screened against the requirements of its own "
            "technology's standard, read from the sections of that standard's own document."
        )
    return read


@pytest.fixture(scope="session")
def invented_standards():
    from config import config

    invented = {
        config.EV_CHARGE_POINT: [
            (config.REGISTERED_EQUIPMENT_REQUIREMENT, "Registered equipment",
             "The equipment is listed in the approved register."),
            (config.RATED_CURRENT_REQUIREMENT, "Rated current",
             "The rated current is at most 28 amps per phase."),
            (config.EARTHING_PROTECTION_REQUIREMENT, "Earthing protection",
             "On a PME earth the equipment has open-PEN protection, and on a TT earth its earth "
             "electrode is at most 165 ohms."),
            (config.SMART_CHARGING_REQUIREMENT, "Smart charging",
             "The equipment has smart charging functionality.")],
        config.HEAT_PUMP: [
            (config.REGISTERED_EQUIPMENT_REQUIREMENT, "Registered equipment",
             "The equipment is listed in the approved register."),
            (config.SINGLE_CONTROLLER_REQUIREMENT, "Single controller",
             "The equipment and every back-up or boost heater are run by a single controller."),
            (config.RATED_CURRENT_REQUIREMENT, "Rated current",
             "The rated current, back-up and boost heaters included, is at most 21.5 amps per "
             "phase."),
            (config.POWER_FACTOR_REQUIREMENT, "Power factor",
             "The power factor is at least 0.92 at full load."),
            (config.EARTHING_PROTECTION_REQUIREMENT, "Earthing protection",
             "On a PME earth the equipment has open-PEN protection, and on a TT earth its earth "
             "electrode is at most 140 ohms.")],
        config.VEHICLE_TO_GRID: [
            (config.REGISTERED_EQUIPMENT_REQUIREMENT, "Registered equipment",
             "The equipment is listed in the type-test register."),
            (config.SMART_CHARGING_REQUIREMENT, "Smart charging",
             "The equipment has smart charging functionality."),
            (config.RATED_CURRENT_REQUIREMENT, "Rated current",
             "The rated current is at most 26 amps per phase."),
            (config.LOSS_OF_MAINS_REQUIREMENT, "Loss-of-mains protection",
             "Loss-of-mains protection is set to the type-test settings."),
            (config.EXPORT_LIMITATION_REQUIREMENT, "Export limitation",
             "An export limitation scheme, where one is fitted, is certified and limits export "
             "to at most 13 amps per phase.")],
        config.SOLAR_PV: [
            (config.REGISTERED_EQUIPMENT_REQUIREMENT, "Registered equipment",
             "The equipment is listed in the type-test register."),
            (config.ISOLATION_REQUIREMENT, "Lockable isolator",
             "An isolator that locks off is fitted on the inverter's mains side."),
            (config.EXPORT_LIMITATION_REQUIREMENT, "Export limitation",
             "An export limitation scheme, where one is fitted, is certified and limits export "
             "to at most 11.5 amps per phase."),
            (config.POWER_FACTOR_REQUIREMENT, "Power factor",
             "The power factor is at least 0.9 at full load.")],
        config.BATTERY_STORAGE: [
            (config.REGISTERED_EQUIPMENT_REQUIREMENT, "Registered equipment",
             "The equipment is listed in the type-test register."),
            (config.CHARGING_CURRENT_REQUIREMENT, "Mains charging current",
             "The charging current from the mains is at most 17 amps per phase."),
            (config.LOSS_OF_MAINS_REQUIREMENT, "Loss-of-mains protection",
             "Loss-of-mains protection is set to the type-test settings."),
            (config.ISOLATION_REQUIREMENT, "Isolation",
             "An isolator that locks off is fitted on the inverter's mains side."),
            (config.EXPORT_LIMITATION_REQUIREMENT, "Export limitation",
             "An export limitation scheme, where one is fitted, is certified and limits export "
             "to at most 14 amps per phase."),
            (config.EARTHING_PROTECTION_REQUIREMENT, "Earthing protection",
             "On a PME earth the equipment has open-PEN protection, and on a TT earth its earth "
             "electrode is at most 95.5 ohms.")],
    }
    documents = []
    for position, (technology, requirements) in enumerate(invented.items(), start=1):
        requirement_lines = []
        for number, requirement in enumerate(requirements, start=1):
            requirement_lines.append(str(number) + "\n" + requirement[1] + ". " + requirement[2])
        documents.append({
            "document_name": config.CONNECTION_STANDARD_DOCUMENTS[technology],
            "document_text": (
                "MERROWDALE POWER NETWORKS\nConnection Standard: "
                + config.TECHNOLOGY_LABELS[technology]
                + "\nDocument Standard 9"
                + str(position)
                + " | Issue 1 | Owner: Engineering Standards | Review cycle: every twelve "
                "months\n1. Purpose and scope\nThis standard is written for the checks alone. Its "
                "figures belong to no standard of Merrowdale Power Networks.\n2. What this "
                "technology covers\nThis standard covers the technology it names, wherever it is "
                "connected to a house's supply.\nThe technology's code is "
                + technology
                + ".\n3. Who applies\nAn application is made by the installer who will fit the "
                "equipment.\n4. Requirements\nEvery requirement below must be Met for a "
                "Connections Engineer to approve the connection. Each requirement is stated once, "
                "in one sentence, and an application is screened against all of them, in this "
                "order:\n"
                + "\n".join(requirement_lines)
                + "\nA requirement is never met by a sentence in a statement, and is never "
                "waived.\n5. Reading the requirements\nEach requirement is Met, Not met or Not "
                "documented on the form as the Connection Rules read it.\n"),
        })
    read = {"problem": None, "invented": invented, "documents": documents, "standards": {}}
    try:
        from knowledge.standard_reading import read_connection_standard

        for technology in invented:
            read["standards"][technology] = read_connection_standard(technology, documents)
            standard = read["standards"][technology]
            assert (isinstance(standard, dict) and isinstance(standard.get("requirements"), list)
                    and all(isinstance(requirement, dict)
                            for requirement in standard["requirements"])), (
                "read_connection_standard returned "
                + str(standard)[:200]
                + " for "
                + technology
                + ", where it returns the standard with its requirements, each a record of its "
                "own."
            )
    except (Exception, SystemExit) as error:
        read["problem"] = (
            "Five standards written for the checks, every figure different from Merrowdale's, "
            "could not be read ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). They are laid out as Merrowdale's standards are, and each technology is read "
            "from its own standard among them with the project's own reader."
        )
    return read


@pytest.fixture(scope="session")
def service_client():
    service = {"problem": None}
    try:
        from fastapi.testclient import TestClient

        from api.routes import service_application

        service["client"] = TestClient(service_application, raise_server_exceptions=False)
    except (Exception, SystemExit) as error:
        service["problem"] = (
            "The REST service could not be started ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Every check on sign-in, access, the queue, the workflow and the chat goes "
            "through it."
        )
    return service


@pytest.fixture(scope="session")
def staff(service_client):
    from config import config

    signed_in = {"problem": service_client["problem"]}
    if signed_in["problem"] is not None:
        return signed_in
    try:
        engineers = [account for account in config.SEED_USER_ACCOUNTS
                     if account["role"] == config.ROLE_CONNECTIONS_ENGINEER]
        coordinators = [account for account in config.SEED_USER_ACCOUNTS
                        if account["role"] == config.ROLE_CONNECTIONS_COORDINATOR]
        advisors = [account for account in config.SEED_USER_ACCOUNTS
                    if account["role"] == config.ROLE_CUSTOMER_SERVICE_ADVISOR]
        for staff_key, account in [("opener", engineers[0]), ("decider", engineers[-1]),
                                   ("coordinator", coordinators[0]), ("advisor", advisors[0])]:
            reply = service_client["client"].post(
                "/auth/signin",
                json={"username": account["username"], "password": account["password"]})
            assert reply.status_code == 200, (
                "Signing in as "
                + account["username"]
                + " with the password the configuration seeds returned "
                + str(reply.status_code)
                + ", so no check that needs a signed-in "
                + account["role"]
                + " can run."
            )
            signed_in[staff_key] = dict(reply.json(), headers={
                config.SESSION_TOKEN_HEADER: reply.json()["session_token"]})
    except (Exception, SystemExit) as error:
        signed_in["problem"] = (
            "The seeded staff could not be signed in ("
            + type(error).__name__
            + ": "
            + str(error)[:400]
            + "). The accounts are the ones the configuration seeds through the database "
            "setup, and a sign-in answers with the user's display name, role and a session "
            "token."
        )
    return signed_in


@pytest.fixture(scope="session", autouse=True)
def language_model_requests(request):
    from concurrent.futures import ThreadPoolExecutor

    from config import config

    question = ("What does the heat pump connection standard ask for before a connection is "
                "approved?")
    started = {"review": {"problem": "The review run was never started."},
               "chat": {"problem": "The chat question was never asked.", "question": question,
                        "latest_turn_before": 0}}
    wanted = set()
    for item in request.session.items:
        wanted.update({"saved_run", "chat_replies"} & set(item.fixturenames))
    with ThreadPoolExecutor(max_workers=2) as executor:
        service_client = {"problem": "The REST service was never started."}
        staff = {"problem": None}
        if wanted:
            service_client = request.getfixturevalue("service_client")
            staff = request.getfixturevalue("staff")
        service_ready = service_client["problem"] is None and staff["problem"] is None
        if "saved_run" in wanted and service_ready:
            client = service_client["client"]
            coordinator = staff["coordinator"]
            review = {"problem": None,
                      "coordinator_request": ("Are the requirements met on the form, and does "
                                              "any statement disagree with it?")}
            try:
                queue = client.get("/queue", params={"technology": config.QUEUE_TECHNOLOGY_ALL},
                                   headers=coordinator["headers"])
                assert queue.status_code == 200 and isinstance(queue.json(), list), (
                    "The application queue returned "
                    + str(queue.status_code)
                    + " with "
                    + queue.text[:300]
                    + ", so no application could be chosen to review."
                )
                reviewed = client.get("/reviews",
                                      params={"status_value": config.REVIEW_STATUS_OPTIONS},
                                      headers=coordinator["headers"])
                assert reviewed.status_code == 200, (
                    "Listing the application reviews returned "
                    + str(reviewed.status_code)
                    + " with "
                    + reviewed.text[:300]
                    + ", so an application nobody has reviewed yet could not be chosen."
                )
                already_reviewed = {record["application_reference"] for record in reviewed.json()}
                choices = []
                for queue_row in queue.json():
                    if (queue_row["assessment"] == config.DECISION_APPROVE
                            and queue_row["application_reference"] not in already_reviewed):
                        choices.append((queue_row["document_count"],
                                        queue_row["application_reference"]))
                assert len(choices) > 0, (
                    "No application in the queue whose assessment is APPROVE CONNECTION is left "
                    "unreviewed, so the one review run cannot be opened. Start again from an "
                    "empty database."
                )
                review["application_reference"] = sorted(choices)[0][1]
                reply = client.get("/queue/" + review["application_reference"],
                                   headers=coordinator["headers"])
                assert reply.status_code == 200, (
                    "The detail of "
                    + review["application_reference"]
                    + " returned "
                    + str(reply.status_code)
                    + " with "
                    + reply.text[:300]
                    + ", so what the review starts from cannot be read."
                )
                review["detail"] = reply.json()
                from persistence.database import session_factory
                from persistence.models import AgentRun

                with session_factory() as session:
                    review["runs_before"] = session.query(AgentRun).filter(
                        AgentRun.application_reference == review["application_reference"]).count()
                review["opened"] = executor.submit(
                    client.post, "/reviews",
                    json={"application_reference": review["application_reference"],
                          "coordinator_request": review["coordinator_request"]},
                    headers=staff["opener"]["headers"])
            except (Exception, SystemExit) as error:
                review["problem"] = (
                    "The one review run that the agent, workflow and record checks read could "
                    "not be opened ("
                    + type(error).__name__
                    + ": "
                    + str(error)[:600]
                    + ")."
                )
            started["review"] = review
        if "chat_replies" in wanted and service_ready:
            client = service_client["client"]
            asker = staff["coordinator"]
            chat = {"problem": None, "question": question, "latest_turn_before": 0}
            try:
                earlier = client.get("/chat/turns", params={"turn_limit": 1},
                                     headers=asker["headers"])
                if earlier.status_code == 200:
                    chat["latest_turn_before"] = max([turn["turn_id"] for turn in earlier.json()]
                                                     + [0])
                chat["asked"] = executor.submit(client.post, "/chat/ask",
                                                json={"question": question},
                                                headers=asker["headers"])
            except (Exception, SystemExit) as error:
                chat["problem"] = (
                    "The chat question could not be asked ("
                    + type(error).__name__
                    + ": "
                    + str(error)[:300]
                    + ")."
                )
            started["chat"] = chat
        yield started


@pytest.fixture(scope="session")
def saved_run(language_model_requests, service_client, staff):
    run = {"problem": service_client["problem"] or staff["problem"], "decision_problem": None}
    if run["problem"] is not None:
        return run
    from config import config

    client = service_client["client"]
    coordinator = staff["coordinator"]
    opener = staff["opener"]
    decider = staff["decider"]
    review = language_model_requests["review"]
    run["problem"] = review["problem"]
    if run["problem"] is not None:
        return run
    run["coordinator_request"] = review["coordinator_request"]
    run["application_reference"] = review["application_reference"]
    run["detail"] = review["detail"]
    run["runs_before"] = review["runs_before"]
    run["decision_reason"] = ("Every requirement is met on the form and no network trigger "
                              "applies, so the equipment may be connected as applied.")
    run["opener"] = opener
    run["decider"] = decider
    for name in ["calls_at_gate", "short_reason", "unknown_decision", "decided", "final_calls",
                 "final_record", "second_review", "short_request"]:
        run[name] = {"status": None, "body": "never requested"}
    run["own_decision"] = {"status": None, "body": "never requested", "calls_before": None,
                           "calls_after": None}
    run["run_record"] = {"problem": "never read"}
    run["heard_at_gate"] = {"problem": "never asked"}
    run["heard_after"] = {"problem": "never asked"}
    try:
        opened = review["opened"].result()
        assert opened.status_code == 201, (
            "Opening a review of "
            + run["application_reference"]
            + " returned "
            + str(opened.status_code)
            + " with "
            + opened.text[:400]
            + ", so the workflow never reached a Connections Engineer."
        )
        run["at_gate"] = opened.json()
        review_id = run["at_gate"]["review_id"]
    except (Exception, SystemExit) as error:
        run["problem"] = (
            "The one review run that the agent, workflow and record checks read could not be "
            "opened ("
            + type(error).__name__
            + ": "
            + str(error)[:600]
            + ")."
        )
        return run

    calls_path = "/reviews/" + review_id + "/calls"
    decision_path = "/reviews/" + review_id + "/decision"
    try:
        reply = client.get(calls_path, headers=coordinator["headers"])
        run["calls_at_gate"] = {"status": reply.status_code, "body": reply.json()}
    except (Exception, SystemExit) as error:
        run["calls_at_gate"]["body"] = type(error).__name__ + ": " + str(error)[:300]

    try:
        from chat.record_answer import answer_application_record
        from persistence.database import session_factory
        from tools.application_data_tools import load_open_applications

        open_applications = load_open_applications()
        heard = {"problem": None}
        with session_factory() as session:
            for heard_key, asker, question_type in [
                    ("advisor_status", staff["advisor"], config.CHAT_TYPE_APPLICATION_STATUS),
                    ("advisor_prediction", staff["advisor"], config.CHAT_TYPE_OUTCOME_PREDICTION),
                    ("team", coordinator, config.CHAT_TYPE_APPLICATION_STATUS)]:
                heard[heard_key] = answer_application_record(
                    session, asker, open_applications, question_type, run["application_reference"])
        run["heard_at_gate"] = heard
    except (Exception, SystemExit) as error:
        run["heard_at_gate"] = {"problem": (
            "What is heard about the application while the review waits could not be worked "
            "out (" + type(error).__name__ + ": " + str(error)[:300] + ")."
        )}

    try:
        before = client.get(calls_path, headers=coordinator["headers"]).json()
        reply = client.post(decision_path, json={"decision": config.DECISION_APPROVE,
                                                 "decision_reason": run["decision_reason"]},
                            headers=opener["headers"])
        after = client.get(calls_path, headers=coordinator["headers"]).json()
        run["own_decision"] = {"status": reply.status_code, "body": reply.json(),
                               "calls_before": before.get("call_count"),
                               "calls_after": after.get("call_count")}
        for refusal_key, payload in [
                ("short_reason", {"decision": config.DECISION_APPROVE,
                                  "decision_reason": "Fine."}),
                ("unknown_decision", {"decision": "CONNECT",
                                      "decision_reason": run["decision_reason"]})]:
            reply = client.post(decision_path, json=payload, headers=decider["headers"])
            run[refusal_key] = {"status": reply.status_code, "body": reply.json()}
        reply = client.post(decision_path, json={"decision": config.DECISION_APPROVE,
                                                 "decision_reason": run["decision_reason"]},
                            headers=decider["headers"])
        run["decided"] = {"status": reply.status_code, "body": reply.json()}
        assert reply.status_code == 200, (
            "Approving the review as "
            + decider["username"]
            + ", a Connections Engineer who did not open it, returned "
            + str(reply.status_code)
            + " with "
            + reply.text[:400]
            + "."
        )
    except (Exception, SystemExit) as error:
        run["decision_problem"] = (
            "The decision on the one review run could not be recorded ("
            + type(error).__name__
            + ": "
            + str(error)[:600]
            + ")."
        )

    try:
        reply = client.get(calls_path, headers=coordinator["headers"])
        run["final_calls"] = {"status": reply.status_code, "body": reply.json()}
        reply = client.get("/reviews", params={"status_value": config.REVIEW_STATUS_OPTIONS},
                           headers=coordinator["headers"])
        matching = [record for record in reply.json() if record["review_id"] == review_id]
        run["final_record"] = {"status": reply.status_code, "body": matching}
        reply = client.post("/reviews",
                            json={"application_reference": run["application_reference"],
                                  "coordinator_request": run["coordinator_request"]},
                            headers=coordinator["headers"])
        run["second_review"] = {"status": reply.status_code, "body": reply.json()}
        reply = client.post("/reviews",
                            json={"application_reference": run["application_reference"],
                                  "coordinator_request": "ok"},
                            headers=coordinator["headers"])
        run["short_request"] = {"status": reply.status_code, "body": reply.json()}
    except (Exception, SystemExit) as error:
        run["final_record"]["body"] = type(error).__name__ + ": " + str(error)[:300]

    try:
        from persistence.database import session_factory
        from persistence.models import AgentRun

        with session_factory() as session:
            stored_run = session.get(AgentRun, run["at_gate"]["run_id"])
            run["run_record"] = {"problem": None, "status": stored_run.status,
                                 "llm_call_count": stored_run.llm_call_count,
                                 "thread_id": stored_run.thread_id,
                                 "runs_opened": session.query(AgentRun).filter(
                                     AgentRun.application_reference
                                     == run["application_reference"]).count()
                                 - run["runs_before"]}
    except (Exception, SystemExit) as error:
        run["run_record"] = {"problem": (
            "The run's own record could not be read (" + type(error).__name__ + ": "
            + str(error)[:300] + ")."
        )}

    try:
        from chat.record_answer import answer_application_record
        from persistence.database import session_factory
        from tools.application_data_tools import load_open_applications

        open_applications = load_open_applications()
        heard = {"problem": None}
        with session_factory() as session:
            for heard_key, asker, question_type in [
                    ("advisor_status", staff["advisor"], config.CHAT_TYPE_APPLICATION_STATUS),
                    ("advisor_prediction", staff["advisor"], config.CHAT_TYPE_OUTCOME_PREDICTION),
                    ("team", opener, config.CHAT_TYPE_OUTCOME_PREDICTION)]:
                heard[heard_key] = answer_application_record(
                    session, asker, open_applications, question_type, run["application_reference"])
        run["heard_after"] = heard
    except (Exception, SystemExit) as error:
        run["heard_after"] = {"problem": (
            "What is heard about the application after the decision could not be worked out ("
            + type(error).__name__ + ": " + str(error)[:300] + ")."
        )}
    return run


@pytest.fixture(scope="session")
def chat_replies(language_model_requests, service_client, staff):
    chat = language_model_requests["chat"]
    replies = {"problem": service_client["problem"] or staff["problem"] or chat["problem"],
               "standard_question": chat["question"],
               "standard": {"status": None, "body": "never asked"},
               "turns": {"status": None, "body": "never read"},
               "latest_turn_before": chat["latest_turn_before"]}
    if replies["problem"] is not None:
        return replies
    client = service_client["client"]
    asker = staff["coordinator"]
    replies["asker"] = asker["username"]
    replies["someone_else"] = staff["advisor"]["username"]
    try:
        reply = chat["asked"].result()
        replies["standard"] = {"status": reply.status_code, "body": reply.json()}
        reply = client.get("/chat/turns", params={"turn_limit": 5}, headers=asker["headers"])
        replies["turns"] = {"status": reply.status_code, "body": reply.json()}
    except (Exception, SystemExit) as error:
        replies["standard"]["body"] = type(error).__name__ + ": " + str(error)[:300]
    return replies


def test_cleaning_keeps_latest_revision_once(raw_frames, processed_frames, tmp_path):
    import os

    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset, save_processed_dataset
    from data_pipeline.history_cleaning import clean_history_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    try:
        cleaned_again = clean_history_dataset(
            load_raw_dataset(config.RAW_APPLICATION_HISTORY_EVAL_FILE))
    except Exception as error:
        cleaned_again = error
    assert isinstance(cleaned_again, pd.DataFrame), (
        "clean_history_dataset gave "
        + repr(cleaned_again)[:200]
        + " for "
        + config.RAW_APPLICATION_HISTORY_EVAL_FILE
        + ", where it returns the cleaned table."
    )
    rebuilt_path = str(tmp_path / "application_history_eval_cleaned_again.csv")
    save_processed_dataset(cleaned_again, rebuilt_path)
    assert os.path.isfile(rebuilt_path), (
        "save_processed_dataset wrote nothing to "
        + rebuilt_path
        + ", where it writes the table it is given to the path it is given."
    )
    rebuilt = pd.read_csv(rebuilt_path, keep_default_na=False, na_values=[""], low_memory=False)
    saved = processed_frames[config.CLEAN_APPLICATION_HISTORY_EVAL_FILE]
    assert rebuilt.shape == saved.shape and rebuilt.equals(saved), (
        "Cleaning "
        + config.RAW_APPLICATION_HISTORY_EVAL_FILE
        + " again gives "
        + str(rebuilt.shape)
        + " rows and columns where "
        + config.CLEAN_APPLICATION_HISTORY_EVAL_FILE
        + " holds "
        + str(saved.shape)
        + ", or the same shape with different values. Every check reads the cleaned files, "
        "so they have to be what the cleaning code produces now: run the data pipeline "
        "again after changing it."
    )
    for clean_path in [config.CLEAN_APPLICATION_HISTORY_TRAIN_FILE,
                       config.CLEAN_APPLICATION_HISTORY_EVAL_FILE]:
        clean_frame = processed_frames[clean_path]
        repeated = int(clean_frame[config.APPLICATION_REFERENCE_COLUMN].duplicated().sum())
        assert repeated == 0, (
            clean_path
            + " still holds "
            + str(repeated)
            + " rows whose application reference appears on another row. An application sent "
            "again is one application, and a model fitted on it twice weighs it twice."
        )
        outcomes = sorted(str(value) for value in
                          clean_frame[config.HISTORY_TARGET_COLUMN].dropna().unique())
        unanswered = int(clean_frame[config.HISTORY_TARGET_COLUMN].isna().sum())
        assert unanswered == 0 and outcomes == sorted(config.NETWORK_WORK_TARGET_ENCODING), (
            "The outcome column of "
            + clean_path
            + " holds "
            + str(unanswered)
            + " blanks and the outcomes "
            + str(outcomes[:6])
            + " where every spelling is folded onto "
            + str(sorted(config.NETWORK_WORK_TARGET_ENCODING))
            + " and no row is kept without one."
        )
    raw_frame = raw_frames[config.RAW_APPLICATION_HISTORY_EVAL_FILE]
    answered = ~raw_frame[config.HISTORY_TARGET_COLUMN].str.strip().str.lower().isin(
        config.NULL_TOKENS)
    templates = raw_frame[answered].drop_duplicates(config.APPLICATION_REFERENCE_COLUMN).head(2)
    rows = pd.concat([templates.iloc[[0]]] * 3 + [templates.iloc[[1]]] * 2
                     + [templates.iloc[[0]]] * 2, ignore_index=True)
    rows[config.APPLICATION_REFERENCE_COLUMN] = ["LC-950001", "LCT0950001", "LCT0950001",
                                                 "950002", "950002", "LCT0950003", "LC-950003"]
    rows[config.REVISION_NUMBER_COLUMN] = ["1", "2", "2", "1", "1", "1", "2"]
    rows[config.PLANNED_WORKING_DAYS_COLUMN] = ["11", "12", "12", "13", "13", "14", "15"]
    rows.loc[6, config.HISTORY_TARGET_COLUMN] = "N/A"
    planted_path = str(tmp_path / "planted_revisions.csv")
    rows.iloc[[2, 6, 3, 0, 5, 4, 1]].to_csv(planted_path, index=False)
    try:
        cleaned = clean_history_dataset(load_raw_dataset(planted_path))
    except Exception as error:
        cleaned = error
    needed = [config.APPLICATION_REFERENCE_COLUMN, config.REVISION_NUMBER_COLUMN,
              config.PLANNED_WORKING_DAYS_COLUMN]
    assert isinstance(cleaned, pd.DataFrame) and set(needed) <= set(cleaned.columns), (
        "clean_history_dataset gave "
        + repr(cleaned)[:200]
        + " for seven planted rows, where it returns the cleaned table with the columns "
        + str(needed)
        + "."
    )
    got = sorted(zip(cleaned[config.APPLICATION_REFERENCE_COLUMN].map(str),
                     cleaned[config.REVISION_NUMBER_COLUMN].map(str),
                     cleaned[config.PLANNED_WORKING_DAYS_COLUMN].map(str), strict=True))
    expected = [("LCT0950001", "2", "12"), ("LCT0950002", "1", "13")]
    assert got == expected, (
        "Seven planted rows - an application written LC-950001 at revision 1 and LCT0950001 "
        "at revision 2, its second revision repeated exactly and standing first in the file; "
        "an application written 950002 repeated exactly; and an application answered at "
        "revision 1, written LCT0950003, whose revision 2, written LC-950003, records no "
        "outcome yet - each telling its revisions apart by its planned working days, cleaned "
        "to "
        + str(got)
        + " where they are "
        + str(expected)
        + ". A row repeated exactly is one row, the highest revision is the application, and an "
        "application whose highest revision records no outcome is not kept: an outcome is "
        "never taken from an earlier revision."
    )


def test_currents_become_amps_per_phase(raw_frames, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.history_cleaning import clean_history_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    raw_frame = raw_frames[config.RAW_APPLICATION_HISTORY_EVAL_FILE]
    answered = ~raw_frame[config.HISTORY_TARGET_COLUMN].str.strip().str.lower().isin(
        config.NULL_TOKENS)
    template = raw_frame[answered].drop_duplicates(config.APPLICATION_REFERENCE_COLUMN).head(1)
    rated, charging = config.RATED_CURRENT_COLUMN, config.CHARGING_CURRENT_COLUMN
    demand, setting = config.MAXIMUM_DEMAND_COLUMN, config.LOAD_LIMITER_SETTING_COLUMN
    inverter, existing = config.INVERTER_CAPACITY_COLUMN, config.EXISTING_GENERATION_COLUMN
    export_limit = config.EXPORT_LIMIT_COLUMN
    planted = [(rated, "32", "1", "1", 32.0), (rated, "32A", "1", "1", 32.0),
               (rated, "32 amps", "1", "1", 32.0), (rated, "32.05A", "1", "1", 32.1),
               (rated, "32.04A", "1", "1", 32.0), (rated, "32.15A", "1", "1", 32.2),
               (rated, "4600W", "1", "1", 20.0), (rated, "4,600 W", "1", "1", 20.0),
               (rated, "4.14 kW", "1", "1", 18.0), (rated, "3,680 W", "1", "1", 16.0),
               (rated, "3.71kW", "1", "1", 16.1), (rated, "3714W", "1", "1", 16.1),
               (rated, "7.36kW", "3", "1", 32.0), (rated, "11.04 kW", "3", "3", 16.0),
               (rated, "7.36kW", "3", "3", 10.7), (rated, "16A", "3", "3", 16.0),
               (charging, "4.6kW", "1", "1", 20.0), (charging, "20 amps", "1", "1", 20.0),
               (demand, "13.3kVA", "1", "1", 57.8), (demand, "13.8kVA", "1", "1", 60.0),
               (demand, "41.4kVA", "3", "1", 60.0), (setting, "13.8kVA", "3", "3", 20.0),
               (inverter, "3.68kW", "1", "1", 3.68), (inverter, "3.68 kW", "1", "1", 3.68),
               (inverter, "3.68 kVA", "1", "1", 3.68), (inverter, "3680W", "1", "1", 3.68),
               (inverter, "11,040 W", "3", "3", 11.04), (existing, "0W", "1", "1", 0.0),
               (export_limit, "2760W", "1", "1", 2.76)]
    rows = pd.concat([template] * len(planted), ignore_index=True)
    for position, planted_reading in enumerate(planted):
        rows.loc[position, config.APPLICATION_REFERENCE_COLUMN] = "LCT09" + str(60001 + position)
        rows.loc[position, config.REVISION_NUMBER_COLUMN] = "1"
        rows.loc[position, config.SUPPLY_PHASES_COLUMN] = planted_reading[2]
        rows.loc[position, config.EQUIPMENT_PHASES_COLUMN] = planted_reading[3]
        rows.loc[position, planted_reading[0]] = planted_reading[1]
    planted_path = str(tmp_path / "planted_currents.csv")
    rows.to_csv(planted_path, index=False)
    try:
        cleaned = clean_history_dataset(load_raw_dataset(planted_path))
    except Exception as error:
        cleaned = error
    read_columns = {config.APPLICATION_REFERENCE_COLUMN}.union(
        planted_reading[0] for planted_reading in planted)
    assert isinstance(cleaned, pd.DataFrame) and read_columns <= set(cleaned.columns), (
        "clean_history_dataset gave "
        + repr(cleaned)[:200]
        + " for "
        + str(len(planted))
        + " planted currents and powers, where it returns the cleaned table with the columns "
        + str(sorted(read_columns))
        + "."
    )
    cleaned = cleaned.set_index(config.APPLICATION_REFERENCE_COLUMN)
    wrong = []
    for position, (column_name, written, supply, equipment, wanted) in enumerate(planted):
        reference = "LCT09" + str(60001 + position)
        if reference not in cleaned.index:
            wrong.append(column_name + " " + repr(written) + " lost its row")
            continue
        got = cleaned.loc[reference, column_name]
        if pd.isna(got) or got != wanted:
            wrong.append(column_name + " " + repr(written) + " with the supply on " + supply
                         + " phases and the equipment on " + equipment + " read as " + str(got)
                         + ", not " + str(wanted))
    assert len(wrong) == 0, (
        "Currents and powers are not read as the Connection Rules set out: "
        + "; ".join(wrong[:6])
        + ". A current in amps is rounded half up once to one decimal, so 32.05A is 32.1; a "
        "current written as a power is converted exactly - kilowatts times 1,000, divided by "
        "230 volts and by the phases - and rounded half up once, on the equipment's phases for "
        "the rated and charging currents and on the supply's for the demand and a load limiting "
        "device's setting, so a single-phase 7.36kW heat pump on a three-phase house is 32.0 "
        "amps; and a power is read as kilowatts, watts divided by 1,000."
    )
    for column_name, written in [(rated, "32a"), (rated, "32 A"), (rated, "32amps"),
                                 (rated, "32 amp"), (rated, "4.14 KW"), (rated, "4600w"),
                                 (demand, "13.3kva"), (charging, "20 mA"),
                                 (inverter, "3.68 kw"), (inverter, "3680w"),
                                 (inverter, "3.68kWp"), (export_limit, "3.68")]:
        unreadable = template.copy()
        unreadable[config.REVISION_NUMBER_COLUMN] = "1"
        unreadable[config.SUPPLY_PHASES_COLUMN] = "1"
        unreadable[config.EQUIPMENT_PHASES_COLUMN] = "1"
        unreadable[column_name] = written
        unreadable_path = str(tmp_path / "unreadable_current.csv")
        unreadable.to_csv(unreadable_path, index=False)
        refusal = None
        try:
            clean_history_dataset(load_raw_dataset(unreadable_path))
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and column_name in refusal and written in refusal, (
            "Cleaning a "
            + column_name
            + " written "
            + repr(written)
            + " gave "
            + str(refusal)
            + ". A current or a power in no notation the three portals write - another unit, "
            "other capitals or another spacing - is refused with a ValueError naming the column "
            "and the value as written, never guessed at or left blank."
        )


def test_references_technologies_phases_earthing_canonical(raw_frames, processed_frames,
                                                            tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.history_cleaning import clean_history_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    raw_frame = raw_frames[config.RAW_APPLICATION_HISTORY_EVAL_FILE]
    answered = ~raw_frame[config.HISTORY_TARGET_COLUMN].str.strip().str.lower().isin(
        config.NULL_TOKENS)
    template = raw_frame[answered].drop_duplicates(config.APPLICATION_REFERENCE_COLUMN).head(1)
    evcp, heat_pump, v2g = config.EV_CHARGE_POINT, config.HEAT_PUMP, config.VEHICLE_TO_GRID
    solar, battery = config.SOLAR_PV, config.BATTERY_STORAGE
    tn_c_s, tn_s, tt_earth = config.EARTHING_TN_C_S, config.EARTHING_TN_S, config.EARTHING_TT
    planted = [
        ("LCT0970001", "LCT0970001", "EV charge point", evcp, "single", 1, "single", 1,
         "TN-C-S", tn_c_s),
        ("LC-970002", "LCT0970002", "EV CHARGER", evcp, "1ph", 1, "1ph", 1, "PME", tn_c_s),
        ("970003", "LCT0970003", "EVCP", evcp, "1", 1, "1", 1, "TNCS", tn_c_s),
        ("LCT0970004", "LCT0970004", "Heat pump", heat_pump, "three", 3, "single", 1, "TN-S",
         tn_s),
        ("LC-970005", "LCT0970005", "heat pump", heat_pump, "3ph", 3, "1ph", 1, "SNE", tn_s),
        ("970006", "LCT0970006", "HP", heat_pump, "3", 3, "3", 3, "TNS", tn_s),
        ("LCT0970007", "LCT0970007", "Vehicle-to-grid charge point", v2g, "Three", 3, "3-phase",
         3, "TT", tt_earth),
        ("LC-970008", "LCT0970008", "V2G charger", v2g, "3ph", 3, "3ph", 3, "tt", tt_earth),
        ("970009", "LCT0970009", "V2G", v2g, "1-phase", 1, "1", 1, "pme", tn_c_s),
        ("LCT0970010", "LCT0970010", "Solar PV", solar, "Single", 1, "single", 1, "TN-S", tn_s),
        ("LC-970011", "LCT0970011", "Solar panels", solar, "1ph", 1, "1ph", 1, "sne", tn_s),
        ("970012", "LCT0970012", "pv", solar, "3", 3, "3", 3, "TNS", tn_s),
        ("LCT0970013", "LCT0970013", "Battery storage", battery, "single", 1, "single", 1,
         "TT", tt_earth),
        ("LC-970014", "LCT0970014", "Home battery", battery, "3ph", 3, "1ph", 1, "PME", tn_c_s),
        ("97", "LCT0000097", "BESS", battery, "1", 1, "1", 1, "TT", tt_earth)]
    rows = pd.concat([template] * len(planted), ignore_index=True)
    for position, written in enumerate(planted):
        rows.loc[position, config.APPLICATION_REFERENCE_COLUMN] = written[0]
        rows.loc[position, config.REVISION_NUMBER_COLUMN] = "1"
        rows.loc[position, config.TECHNOLOGY_COLUMN] = written[2]
        rows.loc[position, config.SUPPLY_PHASES_COLUMN] = written[4]
        rows.loc[position, config.EQUIPMENT_PHASES_COLUMN] = written[6]
        rows.loc[position, config.EARTHING_ARRANGEMENT_COLUMN] = written[8]
    planted_path = str(tmp_path / "planted_notations.csv")
    rows.to_csv(planted_path, index=False)
    try:
        cleaned = clean_history_dataset(load_raw_dataset(planted_path))
    except Exception as error:
        cleaned = error
    read_columns = {config.APPLICATION_REFERENCE_COLUMN, config.TECHNOLOGY_COLUMN,
                    config.SUPPLY_PHASES_COLUMN, config.EQUIPMENT_PHASES_COLUMN,
                    config.EARTHING_ARRANGEMENT_COLUMN}
    assert isinstance(cleaned, pd.DataFrame) and read_columns <= set(cleaned.columns), (
        "clean_history_dataset gave "
        + repr(cleaned)[:200]
        + " for fifteen planted notations, where it returns the cleaned table."
    )
    wrong = []
    for written in planted:
        matching = cleaned[cleaned[config.APPLICATION_REFERENCE_COLUMN] == written[1]]
        if len(matching) != 1:
            wrong.append(written[0] + " is kept " + str(len(matching)) + " times as "
                         + written[1])
            continue
        got = [matching[column_name].iloc[0] for column_name in
               [config.TECHNOLOGY_COLUMN, config.SUPPLY_PHASES_COLUMN,
                config.EQUIPMENT_PHASES_COLUMN, config.EARTHING_ARRANGEMENT_COLUMN]]
        wanted = [written[3], written[5], written[7], written[9]]
        if any(pd.isna(value) for value in got) or got != wanted:
            wrong.append(str([written[2], written[4], written[6], written[8]]) + " read as "
                         + str(got))
    assert len(wrong) == 0, (
        "References, technologies, phases and earthing arrangements are not read as the "
        "Connection Rules set out: "
        + "; ".join(wrong[:5])
        + ". A reference written LCT and seven digits, LC- and the number or the number alone "
        "is LCT and the number in seven digits; a technology written in full, in the "
        "installer's words or as its code, in any capitals, is its code; phases written single, "
        "1ph or 1 are 1 and three, 3ph or 3 are 3; and an earthing arrangement written TN-C-S, "
        "PME or TNCS is TN-C-S, TN-S, SNE or TNS is TN-S, and TT is TT."
    )
    for column_name, written in [(config.APPLICATION_REFERENCE_COLUMN, "LCT52118"),
                                 (config.APPLICATION_REFERENCE_COLUMN, "LC52118"),
                                 (config.APPLICATION_REFERENCE_COLUMN, "LC-12345678"),
                                 (config.APPLICATION_REFERENCE_COLUMN, "12345678"),
                                 (config.TECHNOLOGY_COLUMN, "Wind turbine"),
                                 (config.SUPPLY_PHASES_COLUMN, "2"),
                                 (config.SUPPLY_PHASES_COLUMN, "single-phase"),
                                 (config.EQUIPMENT_PHASES_COLUMN, "3 phase"),
                                 (config.EARTHING_ARRANGEMENT_COLUMN, "IT"),
                                 (config.EARTHING_ARRANGEMENT_COLUMN, "TN C S")]:
        unreadable = template.copy()
        unreadable[config.REVISION_NUMBER_COLUMN] = "1"
        unreadable[column_name] = written
        unreadable_path = str(tmp_path / "unreadable_notation.csv")
        unreadable.to_csv(unreadable_path, index=False)
        refusal = None
        try:
            clean_history_dataset(load_raw_dataset(unreadable_path))
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and column_name in refusal and written in refusal, (
            "Cleaning a "
            + column_name
            + " written "
            + repr(written)
            + " gave "
            + str(refusal)
            + ". A value in no notation the three portals write is refused with a ValueError "
            "naming the column and the value as written."
        )
    for clean_path in [config.CLEAN_APPLICATION_HISTORY_TRAIN_FILE,
                       config.CLEAN_APPLICATION_HISTORY_EVAL_FILE,
                       config.CLEAN_CONNECTION_LOGS_TRAIN_FILE,
                       config.CLEAN_CONNECTION_LOGS_EVAL_FILE,
                       config.CLEAN_STATEMENT_SENTENCES_FILE]:
        clean_frame = processed_frames[clean_path]
        references = clean_frame[config.APPLICATION_REFERENCE_COLUMN].dropna().astype(str)
        stray = sorted(references[~references.str.fullmatch(
            config.CANONICAL_APPLICATION_REFERENCE_PATTERN)].unique())
        assert len(references) == len(clean_frame) and len(stray) == 0, (
            clean_path
            + " holds "
            + str(len(clean_frame) - len(references))
            + " blank application references and "
            + str(len(stray))
            + " that are not canonical, for example "
            + str(stray[:3])
            + "."
        )
        if config.TECHNOLOGY_COLUMN in clean_frame.columns:
            technologies = sorted(str(value) for value in
                                  clean_frame[config.TECHNOLOGY_COLUMN].dropna().unique())
            phases = sorted({int(value) for column_name in config.PHASE_COLUMNS
                             for value in clean_frame[column_name].dropna()})
            untyped = int(clean_frame[config.TECHNOLOGY_COLUMN].isna().sum())
            assert (technologies == sorted(config.CONNECTION_STANDARD_DOCUMENTS)
                    and untyped == 0 and phases == [config.SINGLE_PHASE, config.THREE_PHASE]), (
                clean_path
                + " holds the technologies "
                + str(technologies)
                + ", "
                + str(untyped)
                + " applications with no technology and the phases "
                + str(phases)
                + ", where a technology is one of the five codes and phases are 1 or 3."
            )


def test_answers_and_amounts_read_canonically(raw_frames, processed_frames, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.history_cleaning import clean_history_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    for clean_path, value_mappings in [
            (config.CLEAN_APPLICATION_HISTORY_TRAIN_FILE, config.HISTORY_VALUE_MAPPINGS),
            (config.CLEAN_APPLICATION_HISTORY_EVAL_FILE, config.HISTORY_VALUE_MAPPINGS),
            (config.CLEAN_CONNECTION_LOGS_TRAIN_FILE, config.LOG_VALUE_MAPPINGS),
            (config.CLEAN_CONNECTION_LOGS_EVAL_FILE, config.LOG_VALUE_MAPPINGS),
            (config.CLEAN_STATEMENT_SENTENCES_FILE, config.SENTENCE_VALUE_MAPPINGS)]:
        clean_frame = processed_frames[clean_path]
        for column_name, value_mapping in value_mappings.items():
            found = set(clean_frame[column_name].dropna().astype(str))
            stray = sorted(found - set(value_mapping.values()))
            assert len(stray) == 0, (
                column_name
                + " in "
                + clean_path
                + " holds "
                + str(stray[:5])
                + ", which are not among its canonical values "
                + str(sorted(set(value_mapping.values()))[:6])
                + ". Every spelling a portal writes is folded onto one."
            )
    raw_frame = raw_frames[config.RAW_APPLICATION_HISTORY_EVAL_FILE]
    answered = ~raw_frame[config.HISTORY_TARGET_COLUMN].str.strip().str.lower().isin(
        config.NULL_TOKENS)
    template = raw_frame[answered].drop_duplicates(config.APPLICATION_REFERENCE_COLUMN).head(1)
    looped, listed = config.LOOPED_SERVICE_COLUMN, config.REGISTER_LISTED_COLUMN
    urgency, premises = config.URGENCY_COLUMN, config.PREMISES_TYPE_COLUMN
    service, region = config.SERVICE_TYPE_COLUMN, config.REGION_TYPE_COLUMN
    power_factor, electrode = config.POWER_FACTOR_COLUMN, config.EARTH_ELECTRODE_COLUMN
    cut_out, planned = config.CUT_OUT_RATING_COLUMN, config.PLANNED_WORKING_DAYS_COLUMN
    outcome, work_type = config.HISTORY_TARGET_COLUMN, "network_work_type"
    planted = [
        ({looped: "Yes", listed: "No", urgency: "Heating failed", premises: "Detached house",
          service: "Underground", region: "Urban", power_factor: "0.96",
          electrode: "150 ohms", cut_out: "60A", planned: "10",
          outcome: "Connected after network work", work_type: "Unlooping"},
         {looped: config.BOOLEAN_YES, listed: config.BOOLEAN_NO,
          urgency: config.URGENCY_HEATING_FAILED, premises: config.DETACHED_HOUSE,
          service: config.SERVICE_UNDERGROUND, region: config.REGION_URBAN, power_factor: 0.96,
          electrode: 150.0, cut_out: 60, planned: 10,
          outcome: config.OUTCOME_CONNECTED_AFTER_NETWORK_WORK, work_type: config.WORK_UNLOOPING}),
        ({looped: "Y", listed: "N", urgency: "No heating", premises: "Semi", service: "OH",
          region: "Rural area", power_factor: ".96", electrode: "150\u03a9", cut_out: "60 amp",
          planned: "10 WD", outcome: "Connected - network work", work_type: "Cut-out change"},
         {looped: config.BOOLEAN_YES, listed: config.BOOLEAN_NO,
          urgency: config.URGENCY_HEATING_FAILED, premises: config.SEMI_DETACHED_HOUSE,
          service: config.SERVICE_OVERHEAD, region: config.REGION_RURAL, power_factor: 0.96,
          electrode: 150.0, cut_out: 60, planned: 10,
          outcome: config.OUTCOME_CONNECTED_AFTER_NETWORK_WORK,
          work_type: config.WORK_CUT_OUT_UPGRADE}),
        ({looped: "TRUE", listed: "FALSE", urgency: "HTG-FAIL", premises: "TERR",
          service: "U/G", region: "SUB", power_factor: "96%", electrode: "0.15k\u03a9",
          cut_out: "60", planned: "10 working days", outcome: "CONN-NW", work_type: "REINF"},
         {looped: config.BOOLEAN_YES, listed: config.BOOLEAN_NO,
          urgency: config.URGENCY_HEATING_FAILED, premises: config.TERRACED_HOUSE,
          service: config.SERVICE_UNDERGROUND, region: config.REGION_SUBURBAN,
          power_factor: 0.96, electrode: 150.0, cut_out: 60, planned: 10,
          outcome: config.OUTCOME_CONNECTED_AFTER_NETWORK_WORK,
          work_type: config.WORK_NETWORK_REINFORCEMENT}),
        ({looped: "false", listed: "1", urgency: "std", premises: "bung", service: "o/h",
          region: "urb", power_factor: "89.5%", electrode: "0.0687k\u03a9", cut_out: "100A",
          planned: "1 working day", outcome: "connected", work_type: "N/A"},
         {looped: config.BOOLEAN_NO, listed: config.BOOLEAN_YES,
          urgency: config.URGENCY_STANDARD, premises: config.BUNGALOW,
          service: config.SERVICE_OVERHEAD, region: config.REGION_URBAN, power_factor: 0.895,
          electrode: 68.7, cut_out: 100, planned: 1,
          outcome: config.OUTCOME_CONNECTED_AS_APPLIED}),
        ({looped: "n", listed: "y", urgency: "Routine", premises: "FARM", service: "UG",
          region: "Suburban area", power_factor: "1", electrode: "200", cut_out: "45 amp",
          planned: "0", outcome: "Connected - rework"},
         {looped: config.BOOLEAN_NO, listed: config.BOOLEAN_YES,
          urgency: config.URGENCY_STANDARD, premises: config.FARMHOUSE,
          service: config.SERVICE_UNDERGROUND, region: config.REGION_SUBURBAN,
          power_factor: 1.0, electrode: 200.0, cut_out: 45, planned: 0,
          outcome: config.OUTCOME_CONNECTED_AFTER_REWORK})]
    rows = pd.concat([template] * len(planted), ignore_index=True)
    for position, planted_answers in enumerate(planted):
        rows.loc[position, config.APPLICATION_REFERENCE_COLUMN] = "LCT09" + str(80001 + position)
        rows.loc[position, config.REVISION_NUMBER_COLUMN] = "1"
        for column_name, value in planted_answers[0].items():
            rows.loc[position, column_name] = value
    planted_path = str(tmp_path / "planted_answers.csv")
    rows.to_csv(planted_path, index=False)
    try:
        cleaned = clean_history_dataset(load_raw_dataset(planted_path))
    except Exception as error:
        cleaned = error
    read_columns = {config.APPLICATION_REFERENCE_COLUMN}.union(
        *[set(written) for written, wanted in planted])
    assert isinstance(cleaned, pd.DataFrame) and read_columns <= set(cleaned.columns), (
        "clean_history_dataset gave "
        + repr(cleaned)[:200]
        + " for five planted forms, where it returns the cleaned table with the columns "
        + str(sorted(read_columns))
        + "."
    )
    cleaned = cleaned.set_index(config.APPLICATION_REFERENCE_COLUMN)
    wrong = []
    for position, (written, wanted) in enumerate(planted):
        reference = "LCT09" + str(80001 + position)
        if reference not in cleaned.index:
            wrong.append("the planted row " + reference + " was lost")
            continue
        for column_name, expected in wanted.items():
            got = cleaned.loc[reference, column_name]
            if not pd.isna(got) and got == expected:
                continue
            wrong.append(column_name + " " + repr(written[column_name]) + " read as "
                         + repr(got) + ", not " + repr(expected))
    assert len(wrong) == 0, (
        "Answers are not read as the Connection Rules set out: "
        + "; ".join(wrong[:6])
        + ". A named answer takes its one spelling in any capitals; yes is Yes, Y, TRUE or 1 and "
        "no is No, N, FALSE or 0; a power factor written 0.96, .96 or 96% is 0.96, a "
        "percentage divided by 100; a resistance written 150 ohms, 150 and the ohm sign, or "
        "0.15 kilohms is 150 ohms; a cut-out written 60A, 60 amp or 60 is 60; and working days "
        "written 10, 10 WD or 10 working days are 10."
    )
    for column_name, written in [(urgency, "Emergency"), (premises, "Flat"),
                                 (service, "Shared"), (region, "Coastal"), (looped, "Maybe"),
                                 (outcome, "Withdrawn"), (power_factor, "96 %"),
                                 (power_factor, "0,96"), (electrode, "150 ohm"),
                                 (electrode, "0.15K\u03a9"), (cut_out, "60 A"),
                                 (planned, "10 days")]:
        unreadable = template.copy()
        unreadable[config.REVISION_NUMBER_COLUMN] = "1"
        unreadable[column_name] = written
        unreadable_path = str(tmp_path / "unreadable_answer.csv")
        unreadable.to_csv(unreadable_path, index=False)
        refusal = None
        try:
            clean_history_dataset(load_raw_dataset(unreadable_path))
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and column_name in refusal and written in refusal, (
            "Cleaning a "
            + column_name
            + " written "
            + repr(written)
            + " gave "
            + str(refusal)
            + ". A value with no reading in the Connection Rules is refused with a ValueError "
            "naming the column and the value as written."
        )


def test_impossible_readings_blanked_bad_keys_refused(raw_frames, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.connection_log_cleaning import clean_connection_log_dataset
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.history_cleaning import clean_history_dataset
    from data_pipeline.sentence_cleaning import clean_sentence_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    raw_history = raw_frames[config.RAW_APPLICATION_HISTORY_EVAL_FILE]
    answered = ~raw_history[config.HISTORY_TARGET_COLUMN].str.strip().str.lower().isin(
        config.NULL_TOKENS)
    template = raw_history[answered].drop_duplicates(config.APPLICATION_REFERENCE_COLUMN).head(1)
    rows = pd.concat([template] * 5, ignore_index=True)
    for position in range(5):
        rows.loc[position, config.APPLICATION_REFERENCE_COLUMN] = "LCT09" + str(90001 + position)
        rows.loc[position, config.REVISION_NUMBER_COLUMN] = "1"
        rows.loc[position, config.SUPPLY_PHASES_COLUMN] = "1"
        rows.loc[position, config.EQUIPMENT_PHASES_COLUMN] = "1"
    impossible = [(config.RATED_CURRENT_COLUMN, "-32A"), (config.INVERTER_CAPACITY_COLUMN, "-5kW"),
                  (config.POWER_FACTOR_COLUMN, "104%"), (config.CUT_OUT_RATING_COLUMN, "65A"),
                  (config.EARTH_ELECTRODE_COLUMN, "0"), (config.MAXIMUM_DEMAND_COLUMN, "-3kVA")]
    for column_name, written in impossible:
        rows.loc[0, column_name] = written
    rows.loc[1, config.EQUIPMENT_PHASES_COLUMN] = "3"
    rows.loc[1, config.RATED_CURRENT_COLUMN] = "11.04 kW"
    rows.loc[2, config.EQUIPMENT_PHASES_COLUMN] = "3ph"
    rows.loc[2, config.RATED_CURRENT_COLUMN] = "16A"
    blanked = [config.LOOPED_SERVICE_COLUMN, config.RATED_CURRENT_COLUMN,
               config.PREMISES_TYPE_COLUMN, config.POWER_FACTOR_COLUMN]
    for position, blank_tokens in [(3, [" N/A ", "tbc", "unknown", "Not recorded"]),
                                   (4, ["-", "NA", " n/a", "TBC "])]:
        for column_name, blank_token in zip(blanked, blank_tokens, strict=True):
            rows.loc[position, column_name] = blank_token
    planted_path = str(tmp_path / "planted_impossible.csv")
    rows.to_csv(planted_path, index=False)
    try:
        cleaned = clean_history_dataset(load_raw_dataset(planted_path))
    except Exception as error:
        cleaned = error
    read_columns = {config.APPLICATION_REFERENCE_COLUMN, config.SUPPLY_PHASES_COLUMN,
                    config.EQUIPMENT_PHASES_COLUMN}.union(
        [column_name for column_name, written in impossible], blanked)
    assert isinstance(cleaned, pd.DataFrame) and read_columns <= set(cleaned.columns), (
        "clean_history_dataset gave "
        + repr(cleaned)[:200]
        + " for five planted applications, where it returns the cleaned table with the columns "
        + str(sorted(read_columns))
        + "."
    )
    cleaned = cleaned.set_index(config.APPLICATION_REFERENCE_COLUMN)
    planted_references = ["LCT09" + str(90001 + position) for position in range(5)]
    assert sorted(cleaned.index) == planted_references, (
        "Five planted applications cleaned to the rows "
        + str(sorted(cleaned.index))
        + ". A reading that cannot be true, or an answer left blank, never takes its row with "
        "it."
    )
    still_read = [column_name + " " + written for column_name, written in impossible
                  if not pd.isna(cleaned.loc["LCT0990001", column_name])]
    assert len(still_read) == 0, (
        "Readings that cannot be true were kept: "
        + str(still_read)
        + ". A negative current or power, a power factor above 1, a cut-out rating that is not "
        "one of the five sizes and an earth electrode of nought ohms cannot be true, so the "
        "reading is made blank, never corrected, and the row stays."
    )
    phase_rows = [[cleaned.loc[reference, column_name] for column_name in
                   [config.SUPPLY_PHASES_COLUMN, config.EQUIPMENT_PHASES_COLUMN,
                    config.RATED_CURRENT_COLUMN]]
                  for reference in ["LCT0990002", "LCT0990003"]]
    phases_right = (phase_rows[0][0] == 1 and pd.isna(phase_rows[0][1])
                    and pd.isna(phase_rows[0][2]) and phase_rows[1][0] == 1
                    and pd.isna(phase_rows[1][1]) and phase_rows[1][2] == 16.0)
    assert phases_right, (
        "Three-phase equipment on a single-phase supply, rated 11.04 kW and 16A on two planted "
        "applications, cleaned to the supply's phases, the equipment's phases and the rated "
        "current "
        + str(phase_rows)
        + ". Equipment on more phases than its supply cannot be true, so the equipment's phases "
        "are made blank and never the supply's; a rating written as a power then has no phases "
        "to be converted on and is blank too, while a rating written in amps needs none and "
        "stays."
    )
    unblanked = [str(position) + " " + column_name for position in [3, 4]
                 for column_name in blanked
                 if not pd.isna(cleaned.loc["LCT09" + str(90001 + position), column_name])]
    assert len(unblanked) == 0, (
        "An answer written N/A, TBC, unknown, Not recorded, - or NA, with spaces around it or "
        "in any capitals, was read as a value on planted rows "
        + str(unblanked)
        + ". Each portal's way of leaving an answer blank is a blank answer."
    )
    raw_logs = raw_frames[config.RAW_CONNECTION_LOGS_EVAL_FILE]
    log_template = raw_logs.drop_duplicates(
        config.APPLICATION_REFERENCE_COLUMN).head(3).reset_index(drop=True).copy()
    log_rows = log_template.copy()
    log_rows[config.APPLICATION_REFERENCE_COLUMN] = ["LCT0991001", "LCT0991002", "LCT0991003"]
    log_rows[config.LOG_TARGET_COLUMN] = ["-2", "0", "12 working days"]
    log_rows.loc[2, config.PLANNED_WORKING_DAYS_COLUMN] = "-3 WD"
    log_path = str(tmp_path / "planted_logs.csv")
    log_rows.to_csv(log_path, index=False)
    try:
        cleaned_logs = clean_connection_log_dataset(load_raw_dataset(log_path))
    except Exception as error:
        cleaned_logs = error
    assert isinstance(cleaned_logs, pd.DataFrame) and {
            config.APPLICATION_REFERENCE_COLUMN, config.PLANNED_WORKING_DAYS_COLUMN,
            config.LOG_TARGET_COLUMN} <= set(cleaned_logs.columns), (
        "clean_connection_log_dataset gave "
        + repr(cleaned_logs)[:200]
        + " for three planted logs, where it returns the cleaned table."
    )
    kept = cleaned_logs[config.APPLICATION_REFERENCE_COLUMN].astype(str).tolist()
    assert kept == ["LCT0991002", "LCT0991003"] and pd.isna(
            cleaned_logs[config.PLANNED_WORKING_DAYS_COLUMN].iloc[-1]), (
        "Three planted connection logs cleaned to "
        + str(cleaned_logs[[config.APPLICATION_REFERENCE_COLUMN,
                            config.PLANNED_WORKING_DAYS_COLUMN,
                            config.LOG_TARGET_COLUMN]].to_dict("records"))
        + ". Actual working days of -2 cannot be true and are made blank, and being the answer "
        "the log cannot be learned from, so the row goes; nought working days are a reading "
        "like any other; negative planned working days are made blank and the log stays."
    )
    raw_sentences = raw_frames[config.RAW_STATEMENT_SENTENCES_FILE]
    history_keys = pd.concat([template] * 2, ignore_index=True)
    history_keys[config.APPLICATION_REFERENCE_COLUMN] = "LCT0992001"
    history_keys[config.REVISION_NUMBER_COLUMN] = ["1", "1"]
    history_keys[config.PLANNED_WORKING_DAYS_COLUMN] = ["10", "15"]
    blank_revision = template.copy()
    blank_revision[config.APPLICATION_REFERENCE_COLUMN] = "LCT0992002"
    blank_revision[config.REVISION_NUMBER_COLUMN] = "N/A"
    shared_logs = log_template.head(2).copy()
    shared_logs[config.APPLICATION_REFERENCE_COLUMN] = "LC-992003"
    unnamed_logs = log_template.head(2).copy()
    unnamed_logs.loc[unnamed_logs.index[0], config.APPLICATION_REFERENCE_COLUMN] = "-"
    unnamed_sentences = raw_sentences.head(2).copy()
    unnamed_sentences[config.SENTENCE_ID_COLUMN] = ["", "SEN-9920041"]
    shared_sentences = raw_sentences.head(2).copy()
    shared_sentences[config.SENTENCE_ID_COLUMN] = "SEN-9920042"
    for label, cleaner, frame, named in [
            ("two rows of the same application at the same revision", clean_history_dataset,
             history_keys, ["LCT0992001", config.REVISION_NUMBER_COLUMN]),
            ("an application whose revision is left blank", clean_history_dataset,
             blank_revision, ["LCT0992002", config.REVISION_NUMBER_COLUMN]),
            ("two connection logs of one application", clean_connection_log_dataset,
             shared_logs, ["LCT0992003", config.APPLICATION_REFERENCE_COLUMN]),
            ("a connection log with no application reference", clean_connection_log_dataset,
             unnamed_logs, [config.APPLICATION_REFERENCE_COLUMN]),
            ("a sentence with no identifier", clean_sentence_dataset, unnamed_sentences,
             [config.SENTENCE_ID_COLUMN]),
            ("two sentences sharing an identifier", clean_sentence_dataset, shared_sentences,
             ["SEN-9920042", config.SENTENCE_ID_COLUMN])]:
        keyed_path = str(tmp_path / "keyed.csv")
        frame.to_csv(keyed_path, index=False)
        refusal = None
        try:
            cleaner(load_raw_dataset(keyed_path))
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and all(word in refusal for word in named), (
            "Cleaning "
            + label
            + " gave "
            + str(refusal)
            + ". Every application revision, connection log and sentence is named and appears "
            "once: a blank or repeated key is refused with a ValueError naming "
            + " and ".join(named)
            + ", never dropped or numbered quietly."
        )


def test_schemas_leave_out_outcome_and_identifiers(saved_models):
    from config import config

    assert saved_models["problem"] is None, saved_models["problem"]
    network_work_schema = saved_models["network_work_schema"]
    duration_schema = saved_models["duration_schema"]
    for schema_name, schema, configured in [
            ("network work", network_work_schema,
             config.NETWORK_WORK_NUMERIC_FEATURES + config.NETWORK_WORK_CATEGORICAL_FEATURES),
            ("duration", duration_schema,
             config.DURATION_NUMERIC_FEATURES + config.DURATION_CATEGORICAL_FEATURES)]:
        assert len(schema) == len(set(schema)) and sorted(schema) == sorted(configured), (
            "The saved "
            + schema_name
            + " schema names "
            + str(schema)
            + " where the configuration's features are "
            + str(configured)
            + ", each once."
        )
    network_work_never = [config.HISTORY_TARGET_COLUMN, "network_work_type",
                          config.APPLICATION_REFERENCE_COLUMN, config.REVISION_NUMBER_COLUMN,
                          "installer_reference", config.PREMISES_REFERENCE_COLUMN,
                          config.SUBSTATION_REFERENCE_COLUMN, config.SOURCE_PORTAL_COLUMN]
    duration_never = [config.LOG_TARGET_COLUMN, "network_work_needed", "completion_code",
                      config.APPLICATION_REFERENCE_COLUMN, config.SUBSTATION_REFERENCE_COLUMN,
                      config.SOURCE_PORTAL_COLUMN]
    for schema_name, schema, never in [("network work", network_work_schema, network_work_never),
                                       ("duration", duration_schema, duration_never)]:
        leaked = sorted(set(schema) & set(never))
        assert len(leaked) == 0, (
            "The saved "
            + schema_name
            + " schema feeds the model "
            + str(leaked)
            + ". The outcome, an identifier, and anything recorded only once the decision was "
            "taken or the connection was made are never features: a model fed them learns what "
            "it cannot know when the application is reviewed, or learns which application it "
            "is rather than what was asked."
        )
    counts_used = [column_name for column_name in config.REQUIREMENT_COUNT_RESULTS
                   if column_name in network_work_schema]
    flag_columns = list(config.NETWORK_TRIGGER_FLAG_COLUMNS.values())
    flags_used = {schema_name: [column_name for column_name in flag_columns
                                if column_name in schema]
                  for schema_name, schema in [("network work", network_work_schema),
                                              ("duration", duration_schema)]}
    assert (counts_used == list(config.REQUIREMENT_COUNT_RESULTS)
            and flags_used == {"network work": flag_columns, "duration": flag_columns}), (
        "The network work schema uses the requirement counts "
        + str(counts_used)
        + ", and the two schemas use the network trigger flags "
        + str(flags_used)
        + ", where the network work model is fed all three counts - requirements Met, Not met "
        "and Not documented - and both models every network trigger's flag, so they learn from "
        "the screening a live application is shown with."
    )


def test_requirement_counts_and_triggers_use_own_standard(processed_frames, connection_standards):
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import add_network_trigger_flags, add_requirement_counts
    from tools.screening_tools import find_network_triggers, screen_application

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert connection_standards["problem"] is None, connection_standards["problem"]
    standards = connection_standards["standards"]
    history = processed_frames[config.CLEAN_APPLICATION_HISTORY_EVAL_FILE]
    sample = history.groupby(config.TECHNOLOGY_COLUMN, group_keys=False).head(40)
    sample = sample.reset_index(drop=True)
    try:
        counted = add_requirement_counts(sample, standards)
    except Exception as error:
        counted = error
    assert isinstance(counted, pd.DataFrame), (
        "add_requirement_counts gave "
        + repr(counted)[:200]
        + ", where it returns the applications it was given with their requirement counts "
        "beside them."
    )
    missing = [column_name for column_name in config.REQUIREMENT_COUNT_RESULTS
               if column_name not in counted.columns]
    assert len(missing) == 0 and len(counted) == len(sample), (
        "add_requirement_counts returned "
        + str(len(counted))
        + " rows of the "
        + str(len(sample))
        + " it was given, without the columns "
        + str(missing)
        + "."
    )
    assert config.APPLICATION_REFERENCE_COLUMN in counted and list(
            counted[config.APPLICATION_REFERENCE_COLUMN]) == list(
            sample[config.APPLICATION_REFERENCE_COLUMN]), (
        "add_requirement_counts returned the applications in another order, so each row's "
        "counts no longer sit beside its own answers."
    )
    try:
        flagged = add_network_trigger_flags(counted)
    except Exception as error:
        flagged = error
    flag_columns = list(config.NETWORK_TRIGGER_FLAG_COLUMNS.values())
    assert isinstance(flagged, pd.DataFrame) and set(flag_columns) <= set(flagged.columns) and list(
            flagged[config.APPLICATION_REFERENCE_COLUMN]) == list(
            sample[config.APPLICATION_REFERENCE_COLUMN]), (
        "add_network_trigger_flags gave "
        + repr(flagged)[:200]
        + ", where it returns the applications it was given, in their order, with a flag for "
        "each network trigger "
        + str(flag_columns)
        + " beside them."
    )
    wrong = []
    for application, flagged_row in zip(sample.to_dict("records"), flagged.to_dict("records"),
                                        strict=True):
        standard = standards[application[config.TECHNOLOGY_COLUMN]]
        screening = screen_application(application, standard)
        assert isinstance(screening, dict) and set(config.REQUIREMENT_COUNT_RESULTS) <= set(
                screening), (
            "screen_application returned "
            + str(screening)[:200]
            + " for "
            + application[config.APPLICATION_REFERENCE_COLUMN]
            + ", where it returns the screening with its counts "
            + str(list(config.REQUIREMENT_COUNT_RESULTS))
            + "."
        )
        expected = [screening[column_name] for column_name in config.REQUIREMENT_COUNT_RESULTS]
        got = [flagged_row[column_name] for column_name in config.REQUIREMENT_COUNT_RESULTS]
        if got != expected or sum(got) != len(standard["requirements"]):
            wrong.append(application[config.APPLICATION_REFERENCE_COLUMN] + " counted "
                         + str(got) + " where its own standard's screening gives "
                         + str(expected))
        triggers = find_network_triggers(application)
        expected_flags = [int(trigger_name in triggers)
                          for trigger_name in config.NETWORK_TRIGGER_FLAG_COLUMNS]
        got_flags = [flagged_row[column_name] for column_name in flag_columns]
        if got_flags != expected_flags:
            wrong.append(application[config.APPLICATION_REFERENCE_COLUMN] + " flagged "
                         + str(got_flags) + " where its network triggers " + str(triggers)
                         + " give " + str(expected_flags))
    assert len(wrong) == 0, (
        "The requirement counts and the trigger flags disagree with the screening under each "
        "application's own standard: "
        + "; ".join(wrong[:4])
        + ". Each past application is screened against the standard of its own technology, as "
        "a live application is, its requirements Met, Not met and Not documented add up to that "
        "standard's requirements, and each network trigger it meets is flagged 1, the others 0."
    )
    logs = processed_frames[config.CLEAN_CONNECTION_LOGS_EVAL_FILE]
    log_sample = logs.groupby(config.TECHNOLOGY_COLUMN, group_keys=False).head(30).reset_index(
        drop=True)
    try:
        flagged_logs = add_network_trigger_flags(log_sample)
    except Exception as error:
        flagged_logs = error
    assert isinstance(flagged_logs, pd.DataFrame) and set(flag_columns) <= set(
            flagged_logs.columns), (
        "add_network_trigger_flags gave "
        + repr(flagged_logs)[:200]
        + " for connection logs, where the duration features flag the same network triggers."
    )
    log_wrong = [log[config.APPLICATION_REFERENCE_COLUMN]
                 for log, flagged_log in zip(log_sample.to_dict("records"),
                                             flagged_logs.to_dict("records"), strict=True)
                 if [flagged_log[column_name] for column_name in flag_columns]
                 != [int(trigger_name in find_network_triggers(log))
                     for trigger_name in config.NETWORK_TRIGGER_FLAG_COLUMNS]]
    assert len(log_wrong) == 0, (
        "The connection logs "
        + str(log_wrong[:4])
        + " are flagged otherwise than their own network triggers. A connection log is "
        "flagged the way a past application is."
    )
    crossed = dict(standards)
    crossed[config.EV_CHARGE_POINT] = standards[config.HEAT_PUMP]
    charge_point = sample[sample[config.TECHNOLOGY_COLUMN] == config.EV_CHARGE_POINT].head(1)
    refusal = None
    try:
        add_requirement_counts(charge_point, crossed)
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    assert refusal is not None and charge_point[config.APPLICATION_REFERENCE_COLUMN].iloc[
            0] in refusal, (
        "An EV charge point application screened with the heat pump standard handed over under "
        "its technology gave "
        + str(refusal)
        + ". The two standards are twins written from one template, and a standard written for "
        "another technology is refused, naming the application, never used to screen it."
    )
    without_standard = dict(standards)
    del without_standard[config.BATTERY_STORAGE]
    battery = sample[sample[config.TECHNOLOGY_COLUMN] == config.BATTERY_STORAGE].head(1)
    refusal = None
    try:
        add_requirement_counts(battery, without_standard)
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    assert refusal is not None and config.BATTERY_STORAGE in refusal, (
        "A battery storage application with no standard handed over for its technology gave "
        + str(refusal)
        + ". An application whose technology has no standard cannot be screened and is refused, "
        "naming the technology."
    )


def test_inputs_follow_schema_order_with_gaps(processed_frames, saved_models,
                                              connection_standards):
    import numpy as np
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import (add_network_trigger_flags, add_requirement_counts,
                                               build_feature_frame)

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert connection_standards["problem"] is None, connection_standards["problem"]
    try:
        history = add_network_trigger_flags(add_requirement_counts(
            processed_frames[config.CLEAN_APPLICATION_HISTORY_EVAL_FILE].head(60),
            connection_standards["standards"]))
        logs = add_network_trigger_flags(
            processed_frames[config.CLEAN_CONNECTION_LOGS_EVAL_FILE].head(60))
    except Exception as error:
        history = logs = error
    network_work_columns = (config.NETWORK_WORK_NUMERIC_FEATURES
                            + config.NETWORK_WORK_CATEGORICAL_FEATURES)
    duration_columns = config.DURATION_NUMERIC_FEATURES + config.DURATION_CATEGORICAL_FEATURES
    assert (isinstance(history, pd.DataFrame) and set(network_work_columns) <= set(history.columns)
            and isinstance(logs, pd.DataFrame) and set(duration_columns) <= set(logs.columns)), (
        "Adding the requirement counts and the trigger flags gave "
        + repr(history)[:200]
        + ", where it returns the applications with every network work feature "
        + str(network_work_columns)
        + " beside them, and the logs with every duration feature."
    )
    for name, frame, numeric_columns, categorical_columns, schema in [
            ("network work", history, config.NETWORK_WORK_NUMERIC_FEATURES,
             config.NETWORK_WORK_CATEGORICAL_FEATURES, saved_models["network_work_schema"]),
            ("duration", logs, config.DURATION_NUMERIC_FEATURES,
             config.DURATION_CATEGORICAL_FEATURES, saved_models["duration_schema"])]:
        frame = frame.copy()
        for column_name in numeric_columns:
            frame[column_name] = frame[column_name].astype(float)
            frame.loc[frame.index[0], column_name] = float("nan")
        for column_name in categorical_columns:
            frame[column_name] = frame[column_name].astype(object)
            frame.loc[frame.index[1], column_name] = None
        shuffled = frame[list(reversed(list(frame.columns)))]
        features = build_feature_frame(shuffled, numeric_columns, categorical_columns)
        assert isinstance(features, pd.DataFrame), (
            "build_feature_frame returned "
            + type(features).__name__
            + " for the "
            + name
            + " inputs, where it returns a table of the model's inputs."
        )
        assert list(features.columns) == schema, (
            "The "
            + name
            + " inputs are built in the order "
            + str(list(features.columns))
            + " where the saved schema reads "
            + str(schema)
            + ". The model takes its inputs in the order it was trained on, whatever order the "
            "table it is given holds them in."
        )
        assert len(features) == len(frame) and list(features.index) == list(frame.index), (
            "build_feature_frame returned "
            + str(len(features))
            + " rows for "
            + str(len(frame))
            + ", or in another order."
        )
        for column_name in numeric_columns:
            same = np.allclose(features[column_name].to_numpy(dtype=float),
                               frame[column_name].to_numpy(dtype=float), equal_nan=True)
            assert same, (
                column_name
                + " is changed on its way into the "
                + name
                + " model. A number goes in as it was cleaned, a gap as a gap."
            )
        for column_name in categorical_columns:
            gaps = frame[column_name].isna()
            built = features[column_name]
            gaps_as_nan = all(isinstance(value, float) and value != value
                              for value in built[gaps])
            assert gaps_as_nan and list(built[~gaps]) == list(frame.loc[~gaps, column_name]), (
                column_name
                + " is changed on its way into the "
                + name
                + " model: a gap became "
                + str(sorted({repr(value) for value in built[gaps]})[:3])
                + ". An answer in words goes in as cleaned and a gap as NaN, which the model's "
                "own imputer fills, never None, the text 'nan' or an empty string."
            )


def test_models_fit_schema_and_fill_gaps(processed_frames, saved_models, connection_standards):
    import numpy as np
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import (add_network_trigger_flags, add_requirement_counts,
                                               build_feature_frame)

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert connection_standards["problem"] is None, connection_standards["problem"]
    for name, label, schema in [("network_work", "network work",
                                 saved_models["network_work_schema"]),
                                ("duration", "duration", saved_models["duration_schema"])]:
        fitted_on = list(getattr(saved_models[name], "feature_names_in_", []))
        assert fitted_on == schema, (
            "The saved "
            + label
            + " model was fitted on "
            + str(fitted_on)
            + " while its schema names "
            + str(schema)
            + ". A model saved beside a schema it was not fitted on scores live applications on "
            "the wrong columns."
        )
    history_train = processed_frames[config.CLEAN_APPLICATION_HISTORY_TRAIN_FILE]
    try:
        history = add_network_trigger_flags(add_requirement_counts(
            processed_frames[config.CLEAN_APPLICATION_HISTORY_EVAL_FILE].dropna(
                subset=[config.MAXIMUM_DEMAND_COLUMN, config.PREMISES_TYPE_COLUMN]).head(25),
            connection_standards["standards"]))
    except Exception as error:
        history = error
    assert isinstance(history, pd.DataFrame), (
        "Adding the requirement counts and the trigger flags gave "
        + repr(history)[:200]
        + ", where it returns the applications with them beside their answers."
    )
    features = build_feature_frame(history, config.NETWORK_WORK_NUMERIC_FEATURES,
                                   config.NETWORK_WORK_CATEGORICAL_FEATURES)
    log_features = build_feature_frame(
        add_network_trigger_flags(processed_frames[config.CLEAN_CONNECTION_LOGS_EVAL_FILE].dropna(
            subset=[config.PLANNED_WORKING_DAYS_COLUMN, config.REGION_TYPE_COLUMN]).head(25)),
        config.DURATION_NUMERIC_FEATURES, config.DURATION_CATEGORICAL_FEATURES)
    assert isinstance(features, pd.DataFrame) and isinstance(log_features, pd.DataFrame), (
        "build_feature_frame returned "
        + type(features).__name__
        + " and "
        + type(log_features).__name__
        + " for the network work and duration inputs, where it returns a table of the model's "
        "inputs."
    )
    with_gaps = features.copy()
    with_gaps[config.MAXIMUM_DEMAND_COLUMN] = np.nan
    with_gaps[config.PREMISES_TYPE_COLUMN] = np.nan
    filled = features.copy()
    filled[config.MAXIMUM_DEMAND_COLUMN] = float(
        history_train[config.MAXIMUM_DEMAND_COLUMN].median())
    filled[config.PREMISES_TYPE_COLUMN] = history_train[config.PREMISES_TYPE_COLUMN].mode().iloc[0]
    gap_scores = saved_models["network_work"].predict_proba(with_gaps)[:, 1]
    filled_scores = saved_models["network_work"].predict_proba(filled)[:, 1]
    assert np.isfinite(gap_scores).all() and np.allclose(gap_scores, filled_scores), (
        "An application with its maximum demand after connection and its premises type left "
        "blank scores "
        + str(np.round(gap_scores[:3], 4))
        + " where the same application with the training median and the commonest premises "
        "type scores "
        + str(np.round(filled_scores[:3], 4))
        + ". A blank input is filled as training filled it - a number with the median of the "
        "training data, an answer in words with its most common value - so a live application "
        "with a gap is still scored."
    )
    logs_train = processed_frames[config.CLEAN_CONNECTION_LOGS_TRAIN_FILE]
    log_gaps = log_features.copy()
    log_gaps[config.PLANNED_WORKING_DAYS_COLUMN] = np.nan
    log_gaps[config.REGION_TYPE_COLUMN] = np.nan
    log_filled = log_features.copy()
    log_filled[config.PLANNED_WORKING_DAYS_COLUMN] = float(
        logs_train[config.PLANNED_WORKING_DAYS_COLUMN].median())
    log_filled[config.REGION_TYPE_COLUMN] = logs_train[config.REGION_TYPE_COLUMN].mode().iloc[0]
    gap_days = saved_models["duration"].predict(log_gaps)
    filled_days = saved_models["duration"].predict(log_filled)
    assert np.isfinite(gap_days).all() and np.allclose(gap_days, filled_days), (
        "An application with its planned working days and its region type left blank is "
        "expected to take "
        + str(np.round(gap_days[:3], 1))
        + " working days where the same application filled as training filled it gives "
        + str(np.round(filled_days[:3], 1))
        + "."
    )


def test_quality_report_accounts_for_rows(raw_frames, processed_frames):
    import json
    import os

    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.quality_profiler import profile_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert os.path.isfile(config.DATA_QUALITY_REPORT_FILE), (
        "There is no quality report at "
        + config.DATA_QUALITY_REPORT_FILE
        + ". The data pipeline writes it when it cleans the five extracts."
    )
    with open(config.DATA_QUALITY_REPORT_FILE, encoding="utf-8") as report_file:
        report = json.load(report_file)
    datasets = [
        ("application_history_train", config.RAW_APPLICATION_HISTORY_TRAIN_FILE,
         config.CLEAN_APPLICATION_HISTORY_TRAIN_FILE, config.HISTORY_TARGET_COLUMN),
        ("application_history_eval", config.RAW_APPLICATION_HISTORY_EVAL_FILE,
         config.CLEAN_APPLICATION_HISTORY_EVAL_FILE, config.HISTORY_TARGET_COLUMN),
        ("connection_logs_train", config.RAW_CONNECTION_LOGS_TRAIN_FILE,
         config.CLEAN_CONNECTION_LOGS_TRAIN_FILE, config.LOG_TARGET_COLUMN),
        ("connection_logs_eval", config.RAW_CONNECTION_LOGS_EVAL_FILE,
         config.CLEAN_CONNECTION_LOGS_EVAL_FILE, config.LOG_TARGET_COLUMN),
        ("statement_sentences", config.RAW_STATEMENT_SENTENCES_FILE,
         config.CLEAN_STATEMENT_SENTENCES_FILE, config.SENTENCE_TARGET_COLUMN)]
    assert isinstance(report, dict) and sorted(report) == sorted(
            dataset[0] for dataset in datasets) and all(
            isinstance(profile, dict) for profile in report.values()), (
        config.DATA_QUALITY_REPORT_FILE
        + " profiles "
        + str(sorted(report))[:300]
        + " where it profiles each of the five extracts under its own name."
    )
    reasons = ["duplicate_rows", "rows_missing_target", "answered_earlier_revisions_removed",
               "rows_with_impossible_target", "rows_with_too_few_words"]
    counted_keys = ["rows_received", "rows_kept", "rows_removed", "duplicate_rows",
                    "rows_missing_target", "columns"]
    for name, raw_path, clean_path, target_column in datasets:
        profile = report[name]
        try:
            profiled = profile_dataset(load_raw_dataset(raw_path), processed_frames[clean_path],
                                       target_column)
        except Exception as error:
            profiled = error
        assert isinstance(profiled, dict), (
            "profile_dataset gave "
            + repr(profiled)[:200]
            + " for "
            + name
            + ", where it returns the counts that profile the extract."
        )
        expected = {key: profiled.get(key) for key in counted_keys}
        expected.update({
            "rows_received": len(raw_frames[raw_path]),
            "rows_kept": len(processed_frames[clean_path]),
            "rows_removed": len(raw_frames[raw_path]) - len(processed_frames[clean_path]),
            "columns": int(processed_frames[clean_path].shape[1]),
        })
        got = {key: profile.get(key) for key in counted_keys}
        assert got == expected, (
            "The quality report gives "
            + name
            + " as "
            + str(got)
            + " where the files and profile_dataset give "
            + str(expected)
            + ". Rows received are the raw rows and rows kept the cleaned rows, and the report "
            "holds what profile_dataset counts in the extract as the pipeline read it."
        )
        explained = sum(int(profile.get(reason, 0)) for reason in reasons)
        assert explained == expected["rows_removed"], (
            "The quality report's reasons for "
            + name
            + " add up to "
            + str(explained)
            + " rows where "
            + str(expected["rows_removed"])
            + " were removed: "
            + str({reason: profile.get(reason) for reason in reasons if reason in profile})
            + ". Every removed row is a repeat, a row with no answer, an earlier revision, an "
            "answer that cannot be true or a sentence too short to read."
        )
    planted_raw = pd.DataFrame({
        config.APPLICATION_REFERENCE_COLUMN: ["LCT0960001", "LCT0960001", "LCT0960002",
                                              "LCT0960002", "LCT0960003", "LCT0960004"],
        config.PREMISES_TYPE_COLUMN: [config.BUNGALOW, config.BUNGALOW, config.FARMHOUSE,
                                      config.FARMHOUSE, config.DETACHED_HOUSE,
                                      config.TERRACED_HOUSE],
        config.HISTORY_TARGET_COLUMN: [config.OUTCOME_CONNECTED_AS_APPLIED,
                                       config.OUTCOME_CONNECTED_AS_APPLIED, None, None, None,
                                       config.OUTCOME_CONNECTED_AFTER_NETWORK_WORK]})
    planted_clean = planted_raw.iloc[[0, 5]].reset_index(drop=True)
    try:
        profiled = profile_dataset(planted_raw, planted_clean, config.HISTORY_TARGET_COLUMN)
    except Exception as error:
        profiled = error
    expected = {"rows_received": 6, "rows_kept": 2, "rows_removed": 4, "duplicate_rows": 2,
                "rows_missing_target": 2, "columns": 3}
    got = ({key: profiled.get(key) for key in expected} if isinstance(profiled, dict)
           else profiled)
    assert got == expected, (
        "Six planted rows - an application sent twice exactly, an application with no outcome "
        "sent twice exactly, another application with no outcome and one answered - profiled "
        "against the two rows kept as "
        + str(got)
        + " where they are "
        + str(expected)
        + ". A repeat is counted once, beside its first copy, and a row missing its outcome is "
        "counted among the first copies only."
    )


def test_network_work_model_auc_and_calibration(processed_frames, saved_models,
                                                connection_standards):
    import copy

    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from config import config
    from data_pipeline.feature_builder import (add_network_trigger_flags, add_requirement_counts,
                                               build_feature_frame)
    from modeling.model_evaluation import evaluate_classifier

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert connection_standards["problem"] is None, connection_standards["problem"]
    standards = connection_standards["standards"]
    history_train = processed_frames[config.CLEAN_APPLICATION_HISTORY_TRAIN_FILE]
    try:
        held_out = add_network_trigger_flags(add_requirement_counts(
            processed_frames[config.CLEAN_APPLICATION_HISTORY_EVAL_FILE], standards))
        trained_on = add_network_trigger_flags(add_requirement_counts(
            history_train[history_train[config.TECHNOLOGY_COLUMN]
                          == config.HEAT_PUMP].head(3000), standards))
    except Exception as error:
        held_out = trained_on = error
    assert all(isinstance(frame, pd.DataFrame) and config.HISTORY_TARGET_COLUMN in frame
               for frame in [held_out, trained_on]), (
        "Adding the requirement counts and the trigger flags gave "
        + repr(held_out)[:200]
        + ", where it returns the applications with their outcomes, requirement counts and "
        "trigger flags."
    )
    held_out_features = build_feature_frame(held_out, config.NETWORK_WORK_NUMERIC_FEATURES,
                                            config.NETWORK_WORK_CATEGORICAL_FEATURES)
    train_features = build_feature_frame(trained_on, config.NETWORK_WORK_NUMERIC_FEATURES,
                                         config.NETWORK_WORK_CATEGORICAL_FEATURES)
    assert all(isinstance(frame, pd.DataFrame) for frame in [held_out_features, train_features]), (
        "build_feature_frame returned "
        + type(held_out_features).__name__
        + ", where it returns a table of the model's inputs."
    )
    held_out_answers = held_out[config.HISTORY_TARGET_COLUMN].map(
        config.NETWORK_WORK_TARGET_ENCODING)
    train_answers = trained_on[config.HISTORY_TARGET_COLUMN].map(
        config.NETWORK_WORK_TARGET_ENCODING)
    model = saved_models["network_work"]
    scores = model.predict_proba(held_out_features)[:, 1]
    measured = roc_auc_score(held_out_answers, scores)
    assert measured >= config.MINIMUM_NETWORK_WORK_ROC_AUC, (
        "The network work model ranks the held-out applications with a ROC AUC of "
        + str(round(measured, 4))
        + ", below the "
        + str(config.MINIMUM_NETWORK_WORK_ROC_AUC)
        + " the configuration sets. It is scored on the later applications, which it never saw, "
        "with the requirement counts worked out under each application's own standard."
    )
    assert abs(float(scores.mean()) - float(held_out_answers.mean())) <= 0.03, (
        "The network work model's scores average "
        + str(round(float(scores.mean()), 4))
        + " on the held-out applications, where "
        + str(round(float(held_out_answers.mean()), 4))
        + " of them were connected after network work. A network work likelihood is read as the "
        "share of applications like this one that need network work, so the scores average the "
        "share that did."
    )
    evaluation = evaluate_classifier(copy.deepcopy(model), train_features, train_answers,
                                     held_out_features, held_out_answers)
    missing = sorted({"train_roc_auc", "held_out_roc_auc", "generalisation_gap"}
                     - set(evaluation if isinstance(evaluation, dict) else {}))
    assert len(missing) == 0, (
        "evaluate_classifier returned "
        + type(evaluation).__name__
        + " without "
        + str(missing)
        + ", so the network work model's scores cannot be read."
    )
    train_measured = roc_auc_score(train_answers, model.predict_proba(train_features)[:, 1])
    assert (abs(evaluation["held_out_roc_auc"] - measured) < 0.006
            and abs(evaluation["train_roc_auc"] - train_measured) < 0.006
            and abs(evaluation["generalisation_gap"]
                    - (evaluation["train_roc_auc"] - evaluation["held_out_roc_auc"])) < 0.011), (
        "Given the held-out applications and the first training applications of one "
        "technology, the evaluation reports a held-out ROC AUC of "
        + str(evaluation["held_out_roc_auc"])
        + ", a training ROC AUC of "
        + str(evaluation["train_roc_auc"])
        + " and a gap of "
        + str(evaluation["generalisation_gap"])
        + ", where the model's own scores on those rows give "
        + str(round(measured, 4))
        + " and "
        + str(round(train_measured, 4))
        + ". evaluate_classifier scores the fitted model it is given on the rows it is given, "
        "without fitting it again, and the gap is the training score less the held-out one."
    )


def test_duration_model_r2_and_floor(processed_frames, saved_models):
    import copy

    import pandas as pd
    from sklearn.metrics import mean_absolute_error, r2_score

    from config import config
    from data_pipeline.feature_builder import add_network_trigger_flags, build_feature_frame
    from modeling.model_evaluation import evaluate_regressor

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    training_logs = processed_frames[config.CLEAN_CONNECTION_LOGS_TRAIN_FILE]
    try:
        held_out = add_network_trigger_flags(
            processed_frames[config.CLEAN_CONNECTION_LOGS_EVAL_FILE])
        trained_on = add_network_trigger_flags(
            training_logs[training_logs[config.TECHNOLOGY_COLUMN] == config.HEAT_PUMP].head(3000))
    except Exception as error:
        held_out = trained_on = error
    assert all(isinstance(frame, pd.DataFrame) and config.LOG_TARGET_COLUMN in frame
               for frame in [held_out, trained_on]), (
        "add_network_trigger_flags gave "
        + repr(held_out)[:200]
        + ", where it returns the connection logs with their trigger flags beside them."
    )
    held_out_features = build_feature_frame(held_out, config.DURATION_NUMERIC_FEATURES,
                                            config.DURATION_CATEGORICAL_FEATURES)
    train_features = build_feature_frame(trained_on, config.DURATION_NUMERIC_FEATURES,
                                         config.DURATION_CATEGORICAL_FEATURES)
    assert all(isinstance(frame, pd.DataFrame) for frame in [held_out_features, train_features]), (
        "build_feature_frame returned "
        + type(held_out_features).__name__
        + ", where it returns a table of the model's inputs."
    )
    model = saved_models["duration"]
    forecast = model.predict(held_out_features)
    measured = r2_score(held_out[config.LOG_TARGET_COLUMN], forecast)
    assert measured >= config.MINIMUM_DURATION_R2, (
        "The duration model explains the held-out actual working days with an R2 of "
        + str(round(measured, 4))
        + ", below the "
        + str(config.MINIMUM_DURATION_R2)
        + " the configuration sets."
    )
    floor = float(training_logs[config.LOG_TARGET_COLUMN].min())
    assert float(forecast.min()) >= floor, (
        "The duration model expects a held-out connection to take "
        + str(round(float(forecast.min()), 1))
        + " working days, fewer than the "
        + str(floor)
        + " of the quickest connection it learned from. No connection was ever made in fewer, "
        "so a shorter forecast is the model inventing one."
    )
    training_average = float(training_logs[config.LOG_TARGET_COLUMN].mean())
    average_error = mean_absolute_error(held_out[config.LOG_TARGET_COLUMN],
                                        [training_average] * len(held_out))
    model_error = mean_absolute_error(held_out[config.LOG_TARGET_COLUMN], forecast)
    assert model_error < 0.75 * average_error, (
        "The duration model is off by "
        + str(round(model_error, 1))
        + " working days on average, against "
        + str(round(average_error, 1))
        + " for the average duration given to every connection. A forecast that does little "
        "better than the average tells the Connections Engineer nothing about this application."
    )
    evaluation = evaluate_regressor(copy.deepcopy(model), train_features,
                                    trained_on[config.LOG_TARGET_COLUMN], held_out_features,
                                    held_out[config.LOG_TARGET_COLUMN])
    missing = sorted({"train_r2", "held_out_r2", "generalisation_gap"}
                     - set(evaluation if isinstance(evaluation, dict) else {}))
    assert len(missing) == 0, (
        "evaluate_regressor returned "
        + type(evaluation).__name__
        + " without "
        + str(missing)
        + ", so the duration model's scores cannot be read."
    )
    train_measured = r2_score(trained_on[config.LOG_TARGET_COLUMN], model.predict(train_features))
    assert (abs(evaluation["held_out_r2"] - measured) < 0.006
            and abs(evaluation["train_r2"] - train_measured) < 0.006
            and abs(evaluation["generalisation_gap"]
                    - (evaluation["train_r2"] - evaluation["held_out_r2"])) < 0.011), (
        "Given the held-out logs and the first training logs of one technology, "
        "evaluate_regressor reports "
        + str({key: evaluation.get(key) for key in ["train_r2", "held_out_r2",
                                                    "generalisation_gap"]})
        + " where the model's own forecasts on those rows give a held-out R2 of "
        + str(round(measured, 4))
        + " and a training R2 of "
        + str(round(train_measured, 4))
        + ". It scores the fitted model it is given on the rows it is given, without fitting it "
        "again."
    )


def test_network_work_score_rises_with_failures_and_loops(processed_frames, saved_models,
                                                          connection_standards):
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import (add_network_trigger_flags, add_requirement_counts,
                                               build_feature_frame)

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert connection_standards["problem"] is None, connection_standards["problem"]
    try:
        held_out = add_network_trigger_flags(add_requirement_counts(
            processed_frames[config.CLEAN_APPLICATION_HISTORY_EVAL_FILE],
            connection_standards["standards"]))
    except Exception as error:
        held_out = error
    assert isinstance(held_out, pd.DataFrame), (
        "Adding the requirement counts and the trigger flags gave "
        + repr(held_out)[:200]
        + ", where it returns the applications with them beside their answers."
    )
    features = build_feature_frame(held_out, config.NETWORK_WORK_NUMERIC_FEATURES,
                                   config.NETWORK_WORK_CATEGORICAL_FEATURES)
    assert isinstance(features, pd.DataFrame) and set(config.REQUIREMENT_COUNT_RESULTS) <= set(
            features.columns), (
        "build_feature_frame returned "
        + type(features).__name__
        + ", where it returns the model's inputs with the requirement counts among them."
    )
    model = saved_models["network_work"]
    met_column = config.REQUIREMENTS_MET_COUNT_COLUMN
    before = features[features[met_column] >= 1]
    assert len(before) > 50, (
        "Only "
        + str(len(before))
        + " held-out applications are counted with a requirement Met, where the screening finds "
        "far more."
    )
    after = before.copy()
    after[met_column] = after[met_column] - 1
    after[config.REQUIREMENTS_NOT_MET_COUNT_COLUMN] = (
        after[config.REQUIREMENTS_NOT_MET_COUNT_COLUMN] + 1)
    before_scores = model.predict_proba(before)[:, 1]
    after_scores = model.predict_proba(after)[:, 1]
    rising = float((after_scores > before_scores).mean())
    assert rising >= 0.95 and (after_scores.mean() - before_scores.mean()) >= 0.01, (
        "Turning one requirement from Met to Not met raises the network work score for "
        + str(round(100 * rising, 1))
        + "% of the "
        + str(len(before))
        + " held-out applications it applies to, and moves the average from "
        + str(round(float(before_scores.mean()), 4))
        + " to "
        + str(round(float(after_scores.mean()), 4))
        + ". An application that fails more of its standard's requirements was the more often "
        "connected only after network work, so the model must have learned from the screening "
        "counts."
    )
    unlooped = features[features[config.LOOPED_SERVICE_COLUMN] == config.BOOLEAN_NO]
    looped = unlooped.copy()
    looped[config.LOOPED_SERVICE_COLUMN] = config.BOOLEAN_YES
    looped[config.LOOPED_SERVICE_TRIGGER_COLUMN] = 1
    unlooped_scores = model.predict_proba(unlooped)[:, 1]
    looped_scores = model.predict_proba(looped)[:, 1]
    rising = float((looped_scores > unlooped_scores).mean())
    assert len(unlooped) > 50 and rising >= 0.95 and (
            looped_scores.mean() - unlooped_scores.mean()) >= 0.2, (
        "Making the service looped, with its network trigger flagged, raises the network work "
        "score for "
        + str(round(100 * rising, 1))
        + "% of the "
        + str(len(unlooped))
        + " held-out applications on a service that is not looped, and moves the average from "
        + str(round(float(unlooped_scores.mean()), 4))
        + " to "
        + str(round(float(looped_scores.mean()), 4))
        + ". A looped service shares its cable with the house next door and is unlooped before "
        "a large load is connected, so the model must have learned that it needs network work."
    )


def test_expected_days_rise_with_service_and_region(processed_frames, saved_models):
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import add_network_trigger_flags, build_feature_frame

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    try:
        logs = add_network_trigger_flags(processed_frames[config.CLEAN_CONNECTION_LOGS_EVAL_FILE])
    except Exception as error:
        logs = error
    assert isinstance(logs, pd.DataFrame), (
        "add_network_trigger_flags gave "
        + repr(logs)[:200]
        + ", where it returns the connection logs with their trigger flags beside them."
    )
    features = build_feature_frame(logs, config.DURATION_NUMERIC_FEATURES,
                                   config.DURATION_CATEGORICAL_FEATURES)
    assert isinstance(features, pd.DataFrame) and {
            config.LOOPED_SERVICE_COLUMN, config.LOOPED_SERVICE_TRIGGER_COLUMN,
            config.SERVICE_TYPE_COLUMN, config.REGION_TYPE_COLUMN} <= set(features.columns), (
        "build_feature_frame returned "
        + type(features).__name__
        + ", where it returns the duration model's inputs, the looped service, its trigger flag, "
        "the service type and the region type among them."
    )
    model = saved_models["duration"]
    looped_column = config.LOOPED_SERVICE_COLUMN
    flag_column = config.LOOPED_SERVICE_TRIGGER_COLUMN
    for label, shorter, longer, least_days in [
            ("a looped service rather than one of its own",
             features.assign(**{looped_column: config.BOOLEAN_NO, flag_column: 0}),
             features.assign(**{looped_column: config.BOOLEAN_YES, flag_column: 1}), 5.0),
            ("an overhead service rather than an underground one",
             features.assign(**{config.SERVICE_TYPE_COLUMN: config.SERVICE_UNDERGROUND}),
             features.assign(**{config.SERVICE_TYPE_COLUMN: config.SERVICE_OVERHEAD}), 2.0),
            ("a rural region rather than an urban one",
             features.assign(**{config.REGION_TYPE_COLUMN: config.REGION_URBAN}),
             features.assign(**{config.REGION_TYPE_COLUMN: config.REGION_RURAL}), 2.0)]:
        shorter_days = model.predict(shorter)
        longer_days = model.predict(longer)
        lengthened = float((longer_days > shorter_days).mean())
        assert lengthened >= 0.9 and (longer_days.mean() - shorter_days.mean()) >= least_days, (
            "With "
            + label
            + ", the expected working days rise for "
            + str(round(100 * lengthened, 1))
            + "% of the held-out connections and on average from "
            + str(round(float(shorter_days.mean()), 1))
            + " to "
            + str(round(float(longer_days.mean()), 1))
            + " working days. A looped service is unlooped first, an overhead service takes more "
            "work and a rural region more travel, so each makes a connection take longer, and "
            "the model is fed each of them."
        )


def test_bands_and_delay_follow_thresholds(live_files, saved_models, connection_standards):
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import (add_network_trigger_flags, add_requirement_counts,
                                               build_feature_frame)
    from tools.application_data_tools import fetch_application
    from tools.prediction_tools import (describe_expected_working_days,
                                        estimate_expected_working_days,
                                        estimate_network_work_likelihood)
    from tools.screening_tools import screen_application

    assert live_files["problem"] is None, live_files["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert connection_standards["problem"] is None, connection_standards["problem"]
    standards = connection_standards["standards"]
    band_ranges = {
        config.HIGH_LIKELIHOOD_BAND: (float(config.HIGH_LIKELIHOOD_MINIMUM), float("inf")),
        config.MEDIUM_LIKELIHOOD_BAND: (float(config.MEDIUM_LIKELIHOOD_MINIMUM),
                                        float(config.HIGH_LIKELIHOOD_MINIMUM)),
        config.LOW_LIKELIHOOD_BAND: (float("-inf"), float(config.MEDIUM_LIKELIHOOD_MINIMUM))}
    bands = {band: 0 for band in band_ranges}
    given = []
    for application_reference in live_files["applications"][config.APPLICATION_REFERENCE_COLUMN]:
        application = fetch_application(live_files["applications"], application_reference)
        try:
            screening = screen_application(application,
                                           standards[application[config.TECHNOLOGY_COLUMN]])
            likelihood = estimate_network_work_likelihood(
                saved_models["network_work"], saved_models["network_work_schema"], application,
                screening)
            duration = estimate_expected_working_days(
                saved_models["duration"], saved_models["duration_schema"], application, screening)
        except Exception as error:
            screening = likelihood = duration = error
        assert (isinstance(screening, dict)
                and set(config.REQUIREMENT_COUNT_RESULTS) <= set(screening)
                and isinstance(screening.get("network_triggers"), list)
                and isinstance(likelihood, dict)
                and {"network_work_likelihood", "likelihood_band"} <= set(likelihood)
                and isinstance(duration, dict)
                and {"expected_working_days", "planned_working_days", "delay_expected"}
                <= set(duration)
                and isinstance(duration["expected_working_days"], (int, float))), (
            "For "
            + application_reference
            + " the screening, the likelihood and the expected working days came back as "
            + repr(screening)[:120]
            + ", "
            + repr(likelihood)[:120]
            + " and "
            + repr(duration)[:120]
            + ", where the screening carries its counts and network triggers, the likelihood its "
            "figure and band, and the expected working days a number with the planned working "
            "days and the delay."
        )
        given.append((application, screening, likelihood, duration))
    try:
        frame = add_network_trigger_flags(add_requirement_counts(live_files["applications"],
                                                                 standards))
    except Exception as error:
        frame = error
    assert isinstance(frame, pd.DataFrame) and len(frame) == len(given), (
        "add_requirement_counts and add_network_trigger_flags gave "
        + repr(frame)[:200]
        + " for the live queue, where they give its applications back in order, with their "
        "requirement counts and trigger flags, prepared as the models were trained."
    )
    network_work_inputs = build_feature_frame(frame, config.NETWORK_WORK_NUMERIC_FEATURES,
                                              config.NETWORK_WORK_CATEGORICAL_FEATURES)
    duration_inputs = build_feature_frame(frame, config.DURATION_NUMERIC_FEATURES,
                                          config.DURATION_CATEGORICAL_FEATURES)
    assert isinstance(network_work_inputs, pd.DataFrame) and isinstance(duration_inputs,
                                                                        pd.DataFrame), (
        "build_feature_frame returned "
        + type(network_work_inputs).__name__
        + ", where it returns a table of the model's inputs."
    )
    scores = saved_models["network_work"].predict_proba(network_work_inputs)[:, 1]
    forecasts = saved_models["duration"].predict(duration_inputs)
    delays = {True: 0, False: 0}
    for (application, _screening, likelihood, duration), score, forecast in zip(
            given, scores, forecasts, strict=True):
        figure = likelihood["network_work_likelihood"]
        band = likelihood["likelihood_band"]
        lowest, highest = band_ranges.get(band, (None, None))
        assert (isinstance(figure, float) and abs(figure - float(score)) <= 0.005 + 1e-9
                and abs(round(figure * 100) - figure * 100) < 1e-6
                and lowest is not None and lowest <= figure < highest), (
            application[config.APPLICATION_REFERENCE_COLUMN]
            + " is given a network work likelihood of "
            + str(figure)
            + ", band "
            + str(band)
            + ", where its saved model scores it "
            + str(round(float(score), 4))
            + ". The figure is the model's score to "
            + str(config.LIKELIHOOD_DECIMALS)
            + " decimals, and the band is cut on the figure shown: HIGH from "
            + str(config.HIGH_LIKELIHOOD_MINIMUM)
            + ", MEDIUM from "
            + str(config.MEDIUM_LIKELIHOOD_MINIMUM)
            + ", LOW below."
        )
        bands[band] += 1
        planned = application[config.PLANNED_WORKING_DAYS_COLUMN]
        flag = duration["delay_expected"]
        flagged_right = (
            (planned is None and flag is None)
            or (planned is not None and flag is True
                and duration["expected_working_days"] - planned > config.DELAY_MARGIN_WORKING_DAYS)
            or (planned is not None and flag is False
                and duration["expected_working_days"] - planned
                <= config.DELAY_MARGIN_WORKING_DAYS))
        if planned is not None and isinstance(flag, bool):
            delays[flag] += 1
        assert (float(duration["expected_working_days"]).is_integer()
                and abs(duration["expected_working_days"] - float(forecast)) <= 0.5
                and duration["planned_working_days"] == planned
                and flagged_right), (
            application[config.APPLICATION_REFERENCE_COLUMN]
            + " is expected to take "
            + str(duration["expected_working_days"])
            + " working days against "
            + str(planned)
            + " planned, with a delay flagged as "
            + str(duration["delay_expected"])
            + ", where its saved model gives "
            + str(round(float(forecast), 1))
            + ". The expected working days are the model's forecast in whole working days, and a "
            "delay is flagged only when they are more than "
            + str(config.DELAY_MARGIN_WORKING_DAYS)
            + " working days above the planned working days."
        )
        wording = {
            None: config.EXPECTED_DAYS_WITHOUT_PLAN_TEMPLATE.format(
                expected_working_days=duration["expected_working_days"]),
            True: config.DELAY_TEMPLATE.format(
                expected_working_days=duration["expected_working_days"],
                planned_working_days=planned),
            False: config.NO_DELAY_TEMPLATE.format(
                expected_working_days=duration["expected_working_days"],
                planned_working_days=planned)}
        assert describe_expected_working_days(duration) == wording[flag], (
            "The expected working days of "
            + application[config.APPLICATION_REFERENCE_COLUMN]
            + " are described as "
            + repr(describe_expected_working_days(duration))
            + " where the configuration's words for a delay flagged as "
            + str(flag)
            + " give "
            + repr(wording[flag])
            + "."
        )
    assert min(bands.values()) > 0 and min(delays.values()) > 0, (
        "The live queue's likelihoods fall only in the bands "
        + str(bands)
        + " and its delays only in "
        + str(delays)
        + ", so the thresholds cannot all be told apart."
    )
    unplanned = dict(given[0][0])
    unplanned[config.PLANNED_WORKING_DAYS_COLUMN] = None
    unplanned_duration = estimate_expected_working_days(
        saved_models["duration"], saved_models["duration_schema"], unplanned, given[0][1])
    assert (unplanned_duration["delay_expected"] is None
            and unplanned_duration["planned_working_days"] is None
            and describe_expected_working_days(unplanned_duration)
            == config.EXPECTED_DAYS_WITHOUT_PLAN_TEMPLATE.format(
                expected_working_days=unplanned_duration["expected_working_days"])), (
        "An application with no planned working days is described as "
        + repr(describe_expected_working_days(unplanned_duration))
        + " with a delay flagged as "
        + str(unplanned_duration["delay_expected"])
        + ". With nothing planned there is nothing to compare: no delay is worked out."
    )
    first_application, first_screening = given[0][0], given[0][1]
    other_application = given[1][0]
    for label, estimate_name in [("network work likelihood", "likelihood"),
                                 ("expected working days", "duration")]:
        refusal = None
        try:
            if estimate_name == "likelihood":
                estimate_network_work_likelihood(saved_models["network_work"],
                                                 saved_models["network_work_schema"],
                                                 other_application, first_screening)
            else:
                estimate_expected_working_days(saved_models["duration"],
                                               saved_models["duration_schema"],
                                               other_application, first_screening)
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and first_application[
                config.APPLICATION_REFERENCE_COLUMN] in refusal, (
            "A "
            + label
            + " worked out for "
            + other_application[config.APPLICATION_REFERENCE_COLUMN]
            + " with the screening of "
            + first_application[config.APPLICATION_REFERENCE_COLUMN]
            + " gave "
            + str(refusal)
            + ". A screening belongs to one application and is refused for any other, naming it."
        )


def test_sentence_model_macro_f1(processed_frames, saved_models):
    import copy

    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    from config import config
    from modeling.model_evaluation import evaluate_text_classifier

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    sentences = processed_frames[config.CLEAN_STATEMENT_SENTENCES_FILE]
    train_text, held_out_text, train_answers, held_out_answers = train_test_split(
        sentences[config.SENTENCE_TEXT_COLUMN], sentences[config.SENTENCE_TARGET_COLUMN],
        test_size=config.SENTENCE_HELD_OUT_SHARE,
        stratify=sentences[config.SENTENCE_TARGET_COLUMN], random_state=config.RANDOM_SEED)
    model = saved_models["sentence"]
    measured = f1_score(held_out_answers, model.predict(held_out_text), average="macro")
    assert measured >= config.MINIMUM_SENTENCE_MACRO_F1, (
        "The sentence model scored "
        + str(round(measured, 4))
        + " macro F1 on the held-out statement sentences, below the "
        + str(config.MINIMUM_SENTENCE_MACRO_F1)
        + " the configuration sets. Macro F1 weighs the rarest type of sentence as much as the "
        "commonest: an equipment rating read as background is a rating never held to the form."
    )
    from_wiring_notes = (sentences.loc[train_text.index, config.DOCUMENT_TYPE_COLUMN]
                         == config.WIRING_NOTES)
    probe_text = train_text[from_wiring_notes].head(3000)
    probe_answers = train_answers[from_wiring_notes].head(3000)
    evaluation = evaluate_text_classifier(copy.deepcopy(model), probe_text, probe_answers,
                                          held_out_text, held_out_answers)
    missing = sorted({"held_out_macro_f1", "train_macro_f1", "generalisation_gap",
                      "per_class_f1"} - set(evaluation if isinstance(evaluation, dict) else {}))
    assert len(missing) == 0, (
        "evaluate_text_classifier returned "
        + type(evaluation).__name__
        + " without "
        + str(missing)
        + ", so the sentence model's scores cannot be read."
    )
    trained_on = f1_score(probe_answers, model.predict(probe_text), average="macro")
    assert (abs(evaluation["held_out_macro_f1"] - measured) < 0.006
            and abs(evaluation["train_macro_f1"] - trained_on) < 0.006
            and abs(evaluation["generalisation_gap"]
                    - (evaluation["train_macro_f1"] - evaluation["held_out_macro_f1"])) < 0.011), (
        "Given the held-out sentences and the first training sentences of wiring notes, the "
        "evaluation reports macro F1 "
        + str(evaluation["held_out_macro_f1"])
        + " held out and "
        + str(evaluation["train_macro_f1"])
        + " on training, with a gap of "
        + str(evaluation["generalisation_gap"])
        + ", but the model's own predictions give "
        + str(round(measured, 4))
        + " and "
        + str(round(trained_on, 4))
        + ". evaluate_text_classifier scores the fitted model it is given on the sentences it "
        "is given, without fitting it again."
    )


def test_sentence_predictions_reach_every_type(processed_frames, saved_models):
    from sklearn.model_selection import train_test_split

    from config import config

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    vocabulary = sorted(set(config.SENTENCE_TYPE_MAPPING.values()))
    sentences = processed_frames[config.CLEAN_STATEMENT_SENTENCES_FILE]
    found_types = sorted(str(value) for value in
                         sentences[config.SENTENCE_TARGET_COLUMN].dropna().unique())
    unlabelled = int(sentences[config.SENTENCE_TARGET_COLUMN].isna().sum())
    assert unlabelled == 0 and found_types == vocabulary, (
        "The cleaned sentences hold "
        + str(unlabelled)
        + " with no label, and are labelled "
        + str(found_types)
        + " where the sentence types are "
        + str(vocabulary)
        + "."
    )
    model = saved_models["sentence"]
    learned = sorted(str(label) for label in getattr(model, "classes_", []))
    assert learned == vocabulary, (
        "The sentence model knows the classes "
        + str(learned)
        + " where the sentence types are "
        + str(vocabulary)
        + ". A type of sentence the model cannot predict is a type no Connections Engineer ever "
        "sees."
    )
    split = train_test_split(
        sentences[config.SENTENCE_TEXT_COLUMN], sentences[config.SENTENCE_TARGET_COLUMN],
        test_size=config.SENTENCE_HELD_OUT_SHARE,
        stratify=sentences[config.SENTENCE_TARGET_COLUMN], random_state=config.RANDOM_SEED)
    held_out_text, held_out_answers = split[1], split[3]
    predicted = [str(label) for label in model.predict(held_out_text)]
    stray = sorted(set(predicted) - set(vocabulary))
    unreached = sorted(set(vocabulary) - set(predicted))
    assert len(stray) == 0 and len(unreached) == 0, (
        "On the held-out sentences the sentence model predicts "
        + str(stray)
        + " outside the vocabulary and never predicts "
        + str(unreached)
        + "."
    )
    for sentence_type in vocabulary:
        shown = held_out_text[held_out_answers == sentence_type]
        right = sum(1 for label in model.predict(shown) if str(label) == sentence_type)
        assert right >= 0.75 * len(shown), (
            "Of the "
            + str(len(shown))
            + " held-out sentences labelled "
            + sentence_type
            + ", the model reads "
            + str(right)
            + " as that type. Every type of statement sentence is told apart, not only the "
            "common ones."
        )
    unseen = [str(label) for label in model.predict(["Zzq wvx plorth.",
                                                     "No words of this sort were ever written."])]
    assert all(label in vocabulary for label in unseen), (
        "Sentences of words never seen in training were read as "
        + str(unseen)
        + ", outside the sentence types."
    )


def test_statement_text_cleaning_leaves_plain_words(processed_frames):
    import html

    import pandas as pd

    from config import config
    from data_pipeline.common_cleaning import clean_statement_text

    assert processed_frames["problem"] is None, processed_frames["problem"]
    sentences = processed_frames[config.CLEAN_STATEMENT_SENTENCES_FILE]
    texts = sentences[config.SENTENCE_TEXT_COLUMN].map(str)
    left_markup = int(texts.str.contains(r"</?[A-Za-z][A-Za-z0-9]*[ />]", regex=True).sum())
    left_entities = int(texts.str.contains(r"&[#a-z0-9]+;|\xa0", regex=True).sum())
    doubled_spaces = int(texts.str.contains(r"\s{2,}|^\s|\s$", regex=True).sum())
    assert left_markup == 0 and left_entities == 0 and doubled_spaces == 0, (
        config.CLEAN_STATEMENT_SENTENCES_FILE
        + " still holds "
        + str(left_markup)
        + " sentences with markup, "
        + str(left_entities)
        + " with escaped characters and "
        + str(doubled_spaces)
        + " with stray spaces. A statement sentence is read as plain words."
    )
    lengths = sentences[config.SENTENCE_LENGTH_COLUMN]
    miscounted = int((lengths != texts.str.len()).sum())
    assert miscounted == 0, (
        str(miscounted)
        + " cleaned sentences carry a length that is not the length of their cleaned text. The "
        "length is counted again once the markup is gone."
    )
    plain = texts[texts.str.contains("[A-Z]", regex=True)
                  & texts.str.contains("[a-z]", regex=True)
                  & ~texts.str.contains("[&<>]", regex=True)].head(40).reset_index(drop=True)
    expected = clean_statement_text(plain)
    assert isinstance(expected, pd.Series) and len(expected) == len(plain), (
        "clean_statement_text returned "
        + type(expected).__name__
        + " for "
        + str(len(plain))
        + " plain sentences, where it returns a series of the same length, one cleaned text "
        "for each."
    )
    variants = {
        "wrapped in markup": "<div>" + plain.str.replace(" ", " <br/> ", n=1, regex=False)
                             + "</span>",
        "with its characters escaped": plain.map(html.escape).str.replace(" ", "&nbsp;", n=1,
                                                                          regex=False),
        "with no-break and extra spaces": "  " + plain.str.replace(" ", "\xa0 ", n=1,
                                                                   regex=False).str.replace(
            " ", "   ", regex=False) + " ",
    }
    for label, written in variants.items():
        cleaned = clean_statement_text(pd.Series(written))
        assert isinstance(cleaned, pd.Series) and len(cleaned) == len(plain), (
            "clean_statement_text returned "
            + type(cleaned).__name__
            + " for sentences "
            + label
            + ", where it returns a series of the same length."
        )
        differing = [(written.iloc[position], cleaned.iloc[position])
                     for position in range(len(plain))
                     if cleaned.iloc[position] != expected.iloc[position]]
        assert len(differing) == 0, (
            "A sentence "
            + label
            + " cleans to something else than the plain sentence does, for example "
            + str(differing[:2])
            + ". Markup, character entities, no-break spaces and runs of spaces are taken out, "
            "leaving the words the writer meant."
        )
    shouted = clean_statement_text(plain.str.upper())
    assert isinstance(shouted, pd.Series) and len(shouted) == len(plain), (
        "clean_statement_text returned "
        + type(shouted).__name__
        + " for sentences written in capitals, where it returns a series of the same length."
    )
    differing = [(plain.iloc[position], shouted.iloc[position]) for position in range(len(plain))
                 if shouted.iloc[position] != str(expected.iloc[position]).lower()]
    assert len(differing) == 0, (
        "A sentence written in capitals cleans to something other than the plain sentence in "
        "small letters, for example "
        + str(differing[:2])
        + ". Capitals carry no meaning the model was trained on, and the case cannot be "
        "restored further than small letters."
    )
    kept_case = clean_statement_text(plain)
    assert list(kept_case) == list(plain), (
        "A plain sentence written in mixed case is changed by cleaning, for example "
        + str([(plain.iloc[position], kept_case.iloc[position]) for position in range(len(plain))
               if kept_case.iloc[position] != plain.iloc[position]][:2])
        + ". Only a sentence written wholly in capitals is lowered."
    )


def test_statements_split_into_ordered_sentences(live_files, saved_models):
    import re

    from config import config
    from tools.application_data_tools import fetch_application, fetch_application_documents
    from tools.evidence_tools import read_document_evidence

    assert live_files["problem"] is None, live_files["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    vocabulary = set(config.SENTENCE_TYPE_MAPPING.values())
    applications = live_files["applications"]
    documents_file = live_files["documents"]
    read_applications = 0
    for application_reference in list(applications[config.APPLICATION_REFERENCE_COLUMN])[:8]:
        application = fetch_application(applications, application_reference)
        documents = fetch_application_documents(documents_file, application)
        assert isinstance(documents, list) and all(
                isinstance(document, dict) and {config.DOCUMENT_ID_COLUMN,
                                                config.DOCUMENT_TYPE_COLUMN,
                                                config.DOCUMENT_TEXT_COLUMN} <= set(document)
                for document in documents), (
            "fetch_application_documents returned "
            + str(documents)[:200]
            + " for "
            + application_reference
            + ", where it returns the statements attached to the application, each with its "
            "identifier, type and text."
        )
        attached = set(documents_file.loc[documents_file[config.APPLICATION_REFERENCE_COLUMN]
                                          == application_reference, config.DOCUMENT_ID_COLUMN])
        fetched = [document[config.DOCUMENT_ID_COLUMN] for document in documents]
        assert set(fetched) == attached and len(fetched) == len(attached) and fetched == sorted(
                fetched), (
            "The statements of "
            + application_reference
            + " were fetched as "
            + str(fetched)
            + " where the statements attached to it are "
            + str(sorted(attached))
            + ". Every statement attached to an application is fetched once, in the order of the "
            "document identifiers, whatever order they stand in the file."
        )
        try:
            findings = read_document_evidence(saved_models["sentence"], documents)
        except Exception as error:
            findings = error
        assert isinstance(findings, list) and all(
                isinstance(finding, dict) and {"document_id", "document_type", "sentence",
                                               "sentence_type"} <= set(finding)
                and isinstance(finding["sentence"], str) for finding in findings), (
            "read_document_evidence gave "
            + repr(findings)[:200]
            + " for the statements of "
            + application_reference
            + ", where it returns one finding for each sentence, with its statement's identifier "
            "and type, the sentence and the type the model reads in it."
        )
        read_applications += 1
        runs = [finding["document_id"] for position, finding in enumerate(findings)
                if position == 0 or findings[position - 1]["document_id"] != finding["document_id"]]
        assert runs == fetched, (
            "The sentences of "
            + application_reference
            + " come from the statements in the runs "
            + str(runs)
            + " where the statements run "
            + str(fetched)
            + ". Each statement's sentences are read together, in the order written, and the "
            "statements in their own order."
        )
        for document in documents:
            own = [finding["sentence"] for finding in findings
                   if finding["document_id"] == document[config.DOCUMENT_ID_COLUMN]]
            assert " ".join(own) == document[config.DOCUMENT_TEXT_COLUMN], (
                "The sentences read from "
                + document[config.DOCUMENT_ID_COLUMN]
                + " join back to "
                + repr(" ".join(own)[:200])
                + " instead of the statement's own text. A sentence is kept whole, as written."
            )
            unsplit = [sentence for sentence in own
                       if re.search(config.STATEMENT_SENTENCE_PATTERN, sentence)]
            cut_short = ([sentence for sentence in own[:-1]
                          if not sentence.endswith((".", "!", "?"))]
                         + [sentence for sentence in own[1:] if not re.match("[A-Za-z0-9]",
                                                                             sentence)])
            assert len(unsplit) == 0 and len(cut_short) == 0, (
                "The statement "
                + document[config.DOCUMENT_ID_COLUMN]
                + " was split into pieces that still hold the end of a sentence, "
                + str(unsplit[:2])
                + ", or that were cut where no sentence ends, "
                + str(cut_short[:2])
                + ". A sentence ends at a full stop, question mark or exclamation mark followed "
                "by a space and the next word, and only there."
            )
        for finding in findings:
            assert (finding.get("sentence_type") in vocabulary
                    and finding.get("document_type") == [
                        document[config.DOCUMENT_TYPE_COLUMN] for document in documents
                        if document[config.DOCUMENT_ID_COLUMN] == finding["document_id"]][0]), (
                "A sentence of "
                + str(finding.get("document_id"))
                + " is read as "
                + repr(finding.get("sentence_type"))
                + " in a statement of the type "
                + repr(finding.get("document_type"))
                + ". Each sentence carries the type the model reads in it and the type of the "
                "statement it comes from."
            )
    assert read_applications == 8, "Fewer than eight live applications could be read."
    assert read_document_evidence(saved_models["sentence"], []) == [], (
        "An application with no statement attached gives statement evidence. With nothing "
        "attached, nothing is read."
    )
    lowered_text = ("the charge point draws 32 a when charging. it is listed in the approved "
                    "register. the isolator next to it locks off.")
    lowered = read_document_evidence(saved_models["sentence"], [
        {config.DOCUMENT_ID_COLUMN: "DOC-990001", config.DOCUMENT_TYPE_COLUMN: config.WIRING_NOTES,
         config.DOCUMENT_TEXT_COLUMN: lowered_text}])
    pieces = ([finding.get("sentence") for finding in lowered if isinstance(finding, dict)]
              if isinstance(lowered, list) else [])
    assert len(pieces) == 3 and " ".join(str(piece) for piece in pieces) == lowered_text, (
        "A statement of three sentences written in small letters, as cleaning leaves a "
        "statement written in capitals, was split into "
        + str(pieces)
        + ". A sentence ends at its full stop whatever the case of the next word."
    )
    refusal = None
    try:
        read_document_evidence(saved_models["sentence"], [
            {config.DOCUMENT_ID_COLUMN: "DOC-990002",
             config.DOCUMENT_TYPE_COLUMN: config.DESIGN_STATEMENT,
             config.DOCUMENT_TEXT_COLUMN: None}])
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    assert refusal is not None and "DOC-990002" in refusal, (
        "A statement whose text is blank gave "
        + str(refusal)
        + ". A statement with nothing in it cannot be read and is refused with a ValueError "
        "naming the statement."
    )


def test_sentences_link_requirements_and_ratings(live_files, saved_models, connection_standards):
    from config import config
    from tools.application_data_tools import fetch_application, fetch_application_documents
    from tools.evidence_tools import (find_disagreements, link_evidence_to_requirements,
                                      read_document_evidence)
    from tools.screening_tools import screen_application

    assert live_files["problem"] is None, live_files["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert connection_standards["problem"] is None, connection_standards["problem"]
    standards = connection_standards["standards"]
    applications = live_files["applications"]
    seen_technologies = set()
    templates = {}
    for application_reference in applications[config.APPLICATION_REFERENCE_COLUMN]:
        application = fetch_application(applications, application_reference)
        documents = fetch_application_documents(live_files["documents"], application)
        technology = application[config.TECHNOLOGY_COLUMN]
        if technology in seen_technologies or not documents:
            continue
        seen_technologies.add(technology)
        templates[technology] = application
        try:
            screening = screen_application(application, standards[technology])
            findings = read_document_evidence(saved_models["sentence"], documents)
        except Exception as error:
            screening = findings = error
        assert (isinstance(screening, dict) and isinstance(screening.get("requirements"), list)
                and isinstance(findings, list)
                and all(isinstance(finding, dict) for finding in findings)), (
            "For "
            + application_reference
            + " the screening came back as "
            + repr(screening)[:150]
            + " and the statement evidence as "
            + repr(findings)[:150]
            + ", where the screening holds the standard's requirements and the evidence one "
            "finding for each sentence."
        )
        linked = link_evidence_to_requirements(findings, screening)
        assert isinstance(linked, list) and len(linked) == len(findings), (
            "link_evidence_to_requirements returned "
            + str(linked)[:150]
            + " for the "
            + str(len(findings))
            + " sentences of "
            + application_reference
            + ", where it returns one finding for each."
        )
        own_names = [requirement.get("requirement_name")
                     for requirement in screening["requirements"]]
        for finding, linked_finding in zip(findings, linked, strict=True):
            borne = linked_finding.get("requirements_borne_on")
            assert (isinstance(borne, list) and set(borne) <= set(own_names)
                    and borne == [name for name in own_names if name in borne]
                    and (finding.get("sentence_type") != config.BACKGROUND or borne == [])
                    and linked_finding.get("sentence") == finding.get("sentence")
                    and linked_finding.get("sentence_type") == finding.get("sentence_type")
                    and linked_finding.get("document_id") == finding.get("document_id")), (
                "A sentence of "
                + application_reference
                + " read as "
                + str(finding.get("sentence_type"))
                + " bears on "
                + str(borne)
                + " where the application's own standard, "
                + str(screening.get("standard_document"))
                + ", has the requirements "
                + str(own_names)
                + ". A sentence bears only on requirements its application's own standard has, "
                "in that standard's order, background on none, and keeps its sentence, type and "
                "statement."
            )
    assert seen_technologies == set(config.CONNECTION_STANDARD_DOCUMENTS), (
        "The live applications read covered the technologies "
        + str(sorted(seen_technologies))
        + ", where every technology of the configuration has a live application with a "
        "statement."
    )
    named = {"application_reference": "LCT0950110", "technology": config.HEAT_PUMP,
             "standard_document": "Invented-Linking-Standard.pdf", "requirements": []}
    for kind, name in [(config.POWER_FACTOR_REQUIREMENT, "Full-load power factor"),
                       (config.ISOLATION_REQUIREMENT, "Lock-off switch"),
                       (config.RATED_CURRENT_REQUIREMENT, "Current drawn"),
                       (config.EXPORT_LIMITATION_REQUIREMENT, "Export cap"),
                       (config.REGISTERED_EQUIPMENT_REQUIREMENT, "Listed model"),
                       (config.SINGLE_CONTROLLER_REQUIREMENT, "One controller"),
                       (config.EARTHING_PROTECTION_REQUIREMENT, "Earth fault safety"),
                       (config.CHARGING_CURRENT_REQUIREMENT, "Mains charging"),
                       (config.LOSS_OF_MAINS_REQUIREMENT, "Islanding guard"),
                       (config.SMART_CHARGING_REQUIREMENT, "Smart control")]:
        named["requirements"].append({"requirement_kind": kind, "requirement_name": name,
                                      "requirement_sentence": "An invented requirement: " + name
                                                              + ".",
                                      "result": config.REQUIREMENT_MET, "blank_answers": []})
    wanted = {config.EQUIPMENT_RATING: ["Full-load power factor", "Current drawn",
                                        "Mains charging"],
              config.EQUIPMENT_REGISTRATION: ["Listed model"], config.SUPPLY_ARRANGEMENT: [],
              config.EARTHING: ["Earth fault safety"],
              config.PROTECTION_SETTINGS: ["Islanding guard"],
              config.EXPORT_LIMITATION: ["Export cap"],
              config.LOAD_MANAGEMENT: ["One controller", "Smart control"],
              config.ISOLATION_AND_SWITCHING: ["Lock-off switch"], config.BACKGROUND: []}
    linked = link_evidence_to_requirements(
        [{"document_id": "DOC-990003", "document_type": config.DESIGN_STATEMENT,
          "sentence": "A sentence read as " + sentence_type + ".", "sentence_type": sentence_type}
         for sentence_type in wanted], named)
    got = ({finding.get("sentence_type"): finding.get("requirements_borne_on")
            for finding in linked if isinstance(finding, dict)}
           if isinstance(linked, list) else linked)
    assert got == wanted, (
        "Under a standard whose requirements carry names no Merrowdale standard uses, one "
        "sentence of each type bears on "
        + str(got)
        + " where they bear on "
        + str(wanted)
        + ". Each type of sentence bears on the requirements of the kinds its statement guide "
        "names, found by their kind, in the standard's own order, and a supply arrangement and "
        "background on none."
    )
    refusal = None
    try:
        link_evidence_to_requirements(
            [{"document_id": "DOC-990004", "document_type": config.DESIGN_STATEMENT,
              "sentence": "A sentence of a type nobody configured.", "sentence_type": "Rumour"}],
            named)
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    assert refusal is not None and "Rumour" in refusal, (
        "A sentence of a type the configuration gives no requirements for gave "
        + str(refusal)
        + ". It is refused with a ValueError naming the type."
    )
    rating = config.EQUIPMENT_RATING
    heat_pump_sentences = [
        (rating, "The heat pump draws up to 26 A on a dedicated 32 A circuit."),
        (rating, "The heat output is 8 kW."),
        (rating, "The heat pump draws 25 A with its back-up heater running."),
        (rating, "The unit is rated at 5,750 W."),
        (rating, "Its 8 kW heat output is delivered at full load."),
        (rating, "A 100 A supply fuse and an 80A cut-out serve the house."),
        (rating, "Its buffer tank stores 9.5 kWh of heat."),
        (config.SUPPLY_ARRANGEMENT, "The supply to the house is rated at 100 A.")]
    battery_sentences = [
        (rating, "The battery charges from the mains at up to 20 A."),
        (rating, "The inverter exports up to 21.7 A."),
        (rating, "The inverter is rated at 5 kW."),
        (rating, "The battery discharges at up to 6 kW."),
        (rating, "Its storage is 9.5 kWh.")]
    heat_pump = dict(templates[config.HEAT_PUMP])
    heat_pump.update({config.EQUIPMENT_PHASES_COLUMN: 1, config.CHARGING_CURRENT_COLUMN: None,
                      config.INVERTER_CAPACITY_COLUMN: None})
    battery = dict(templates[config.BATTERY_STORAGE])
    battery.update({config.EQUIPMENT_PHASES_COLUMN: 1, config.RATED_CURRENT_COLUMN: None,
                    config.CHARGING_CURRENT_COLUMN: 20.0, config.INVERTER_CAPACITY_COLUMN: 5.0})
    disagrees, blank = config.DISAGREES_WITH_FORM, config.DESCRIBES_BLANK_ANSWER
    cases = [
        ("a single-phase heat pump rated 25 amps", heat_pump_sentences,
         dict(heat_pump, **{config.RATED_CURRENT_COLUMN: 25.0}), [25.0],
         [(heat_pump_sentences[0][1], [26.0], disagrees),
          (heat_pump_sentences[1][1], [34.8], disagrees)]),
        ("a single-phase heat pump with its rated current blank", heat_pump_sentences,
         dict(heat_pump, **{config.RATED_CURRENT_COLUMN: None}), [],
         [(heat_pump_sentences[0][1], [26.0], blank), (heat_pump_sentences[1][1], [34.8], blank),
          (heat_pump_sentences[2][1], [25.0], blank), (heat_pump_sentences[3][1], [25.0], blank)]),
        ("a single-phase battery charging at 20 amps with a 5 kW inverter", battery_sentences,
         battery, [20.0, 21.7], [(battery_sentences[3][1], [26.1], disagrees)])]
    for meaning, sentences, application, form_amps, expected in cases:
        findings = [{"document_id": "DOC-990005", "document_type": config.DESIGN_STATEMENT,
                     "sentence": sentence, "sentence_type": sentence_type}
                    for sentence_type, sentence in sentences]
        flagged = find_disagreements(findings, application)
        assert isinstance(flagged, list) and all(isinstance(flag, dict) for flag in flagged), (
            "find_disagreements returned "
            + str(flagged)[:200]
            + ", where it returns one flag for each equipment rating sentence it flags."
        )
        got = [(flag.get("sentence"), flag.get("stated_amps"), flag.get("flag"))
               for flag in flagged]
        right_form = all(sorted(flag.get("form_rating_amps") or []) == form_amps
                         and flag.get("document_id") == "DOC-990005" for flag in flagged)
        assert got == expected and right_form, (
            "For "
            + meaning
            + ", the equipment rating sentences were flagged as "
            + str(got)
            + " where the Connection Rules flag "
            + str(expected)
            + ". A figure in amps, watts, kilowatts or kVA is made amps per phase on the "
            "equipment's phases and held to every rating on the form; a figure followed within "
            "two words by a fuse, cut-out, supply, cable, circuit, breaker, heat or thermal is "
            "not read, nor kilowatt-hours; a sentence none of whose figures is a rating on the "
            "form disagrees with it, any figure describes a blank rating, and only equipment "
            "rating sentences are read."
        )


def test_every_document_gives_up_text(document_texts, tmp_path):
    import os
    import shutil

    import pypdf

    from config import config
    from knowledge.pdf_extraction import extract_text_from_directory

    assert document_texts["problem"] is None, document_texts["problem"]
    in_folder = sorted(name for name in os.listdir(config.DOCS_DIRECTORY)
                       if name.lower().endswith(".pdf"))
    documents = document_texts["documents"]
    assert len(documents) > 0 and {"document_name", "document_text"} <= set(documents[0]), (
        "Reading the document folder returned "
        + str(len(documents))
        + " entries"
        + ("" if len(documents) == 0 else ", the first keyed " + str(sorted(documents[0])))
        + ", where every entry names its document and carries its text."
    )
    names = sorted(document["document_name"] for document in documents)
    assert names == in_folder, (
        "Reading the document folder returned "
        + str(names)
        + " where the folder holds "
        + str(in_folder)
        + ". A document never read is a question answered from general knowledge instead."
    )
    for document in documents:
        assert len(document["document_text"].split()) > 250, (
            document["document_name"]
            + " gave up only "
            + str(len(document["document_text"].split()))
            + " words. A PDF that will not give up its text is indexed as an empty passage."
        )
    for required in (list(config.CONNECTION_STANDARD_DOCUMENTS.values())
                     + list(config.SENTENCE_TYPE_GUIDE_DOCUMENTS.values())
                     + config.CHAT_EXCLUDED_DOCUMENTS):
        assert required in names, (
            required
            + " was not read, and the configuration reads it by name."
        )
    folder = tmp_path / "mixed_folder"
    folder.mkdir()
    standard_document = config.CONNECTION_STANDARD_DOCUMENTS[config.SOLAR_PV]
    guide_document = config.SENTENCE_TYPE_GUIDE_DOCUMENTS[config.EARTHING]
    for name in [standard_document, guide_document]:
        shutil.copy(os.path.join(config.DOCS_DIRECTORY, name), str(folder / name))
    (folder / "reading-list.txt").write_text("Not a document of the connections team.")
    (folder / "scan.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    mixed_names = None
    try:
        mixed_names = sorted(entry["document_name"]
                             for entry in extract_text_from_directory(str(folder)))
    except Exception as error:
        mixed_names = type(error).__name__ + ": " + str(error)[:200]
    assert mixed_names == sorted([standard_document, guide_document]), (
        "A folder holding two PDFs, a text file and an image was read as "
        + str(mixed_names)
        + ". Only the PDFs are the documents; anything else in the folder is passed over."
    )
    blank_folder = tmp_path / "blank_page_folder"
    blank_folder.mkdir()
    writer = pypdf.PdfWriter()
    writer.add_page(pypdf.PdfReader(os.path.join(config.DOCS_DIRECTORY,
                                                 standard_document)).pages[0])
    writer.add_blank_page()
    with open(str(blank_folder / "Standard-With-A-Blank-Page.pdf"), "wb") as blank_file:
        writer.write(blank_file)
    empty_folder = tmp_path / "empty_folder"
    empty_folder.mkdir()
    (empty_folder / "notes.txt").write_text("No PDF here.")
    for label, folder_path, named in [
            ("A PDF whose second page gives up no text", blank_folder,
             "Standard-With-A-Blank-Page.pdf"),
            ("A folder with no PDF in it", empty_folder, str(empty_folder))]:
        refusal = None
        try:
            extract_text_from_directory(str(folder_path))
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and named in refusal, (
            label
            + " gave "
            + str(refusal)
            + ". A page that cannot be read may be the one that states a requirement, and a "
            "folder with no document has nothing to answer from, so each is refused with a "
            "ValueError naming the file or the folder."
        )


def test_documents_cut_into_sections_and_passages(document_passages, connection_standards):
    import re

    from config import config
    from knowledge.chunk_splitting import build_chunk_records, split_documents_into_sections

    assert document_passages["problem"] is None, document_passages["problem"]
    assert connection_standards["problem"] is None, connection_standards["problem"]
    documents = document_passages["documents"]
    sections = document_passages["sections"]
    chunks = document_passages["chunks"]
    assert len(chunks) > 0 and {"chunk_id", "chunk_text", "document_name"} <= set(chunks[0]), (
        "Cutting the documents returned "
        + str(len(chunks))
        + " passages"
        + ("" if len(chunks) == 0 else ", the first keyed " + str(sorted(chunks[0])))
        + ", where every passage carries its identifier, its text and its document."
    )
    assert len(sections) > 5 * len(documents), (
        str(len(documents))
        + " documents split into "
        + str(len(sections))
        + " sections. Each document is written in numbered sections, and a section is the unit "
        "the tools read and a question is usually about."
    )
    with_header = [chunk["chunk_id"] for chunk in chunks
                   if re.search(config.DOCUMENT_HEADER_PATTERN, chunk["chunk_text"])]
    assert len(with_header) == 0, (
        "These passages still carry the document header: "
        + str(with_header[:5])
        + ". The header repeats the same words in every document, so it matches every question "
        "equally and pushes the passages that answer it out of the results."
    )
    lengths = [len(chunk["chunk_text"]) for chunk in chunks]
    average = sum(lengths) / len(lengths)
    assert (config.CHUNK_SIZE_CHARACTERS / 5 <= average <= config.CHUNK_SIZE_CHARACTERS
            and max(lengths) <= config.CHUNK_SIZE_CHARACTERS), (
        "Passages average "
        + str(int(average))
        + " characters, the longest "
        + str(max(lengths))
        + ", where the configuration asks for passages of up to "
        + str(config.CHUNK_SIZE_CHARACTERS)
        + ". Passages far longer return a whole document for every question; far shorter lose "
        "the sentence that gave them their meaning."
    )
    identifiers = [chunk["chunk_id"] for chunk in chunks]
    assert len(identifiers) == len(set(identifiers)), (
        "Passage identifiers repeat, so storing them would overwrite one passage with another."
    )
    words_by_document = {document["document_name"]: set(document["document_text"].split())
                         for document in documents}
    for chunk in chunks:
        words = chunk["chunk_text"].split()
        foreign = [word for word in words
                   if word not in words_by_document.get(chunk["document_name"], set())]
        assert len(words) > 0 and len(foreign) <= len(words) // 10, (
            "Passage "
            + str(chunk["chunk_id"])
            + " says it comes from "
            + str(chunk["document_name"])
            + " but "
            + str(len(foreign))
            + " of its "
            + str(len(words))
            + " words are not in that document, for example "
            + str(foreign[:5])
            + ". A passage has to cite the document it was cut from."
        )
    for technology, standard in connection_standards["standards"].items():
        own_chunks = [re.sub(r"\s+", " ", chunk["chunk_text"]) for chunk in chunks
                      if chunk["document_name"] == config.CONNECTION_STANDARD_DOCUMENTS[technology]]
        broken = [requirement["requirement_sentence"] for requirement in standard["requirements"]
                  if not any(requirement["requirement_sentence"] in chunk_text
                             for chunk_text in own_chunks)]
        assert len(broken) == 0, (
            "These requirements of "
            + config.CONNECTION_STANDARD_DOCUMENTS[technology]
            + " are not whole inside any one of its passages: "
            + str(broken[:3])
            + ". A requirement cut across two passages can no longer be read from either."
        )
    header = ("Document Standard 99 | Issue 1 | Owner: Engineering Standards | Review cycle: "
              "every six months")
    lines = []
    for first in "abcdefgh":
        for second in "abcdefg":
            lines.append("Reading " + first + second + " of the overlap check names break "
                         + first + second + ".")
    synthetic = [{"document_name": "Overlap-Check.pdf",
                  "document_text": "MERROWDALE POWER NETWORKS\nOverlap Check\n" + header
                                   + "\n1. Overlap check\n" + " ".join(lines)}]
    synthetic_chunks = build_chunk_records(split_documents_into_sections(synthetic))
    assert isinstance(synthetic_chunks, list) and all(
            isinstance(chunk, dict) and "chunk_text" in chunk for chunk in synthetic_chunks), (
        "Cutting a document of one long section returned "
        + str(synthetic_chunks)[:200]
        + ", where it returns a list of passages, each with its text."
    )
    pieces = [str(chunk["chunk_text"]) for chunk in synthetic_chunks]
    assert len(pieces) >= 3, (
        "One numbered section of "
        + str(len(" ".join(lines)))
        + " characters was cut into "
        + str(len(pieces))
        + " passages, where passages of up to "
        + str(config.CHUNK_SIZE_CHARACTERS)
        + " characters need at least three."
    )
    unshared = [position for position in range(1, len(pieces))
                if pieces[position].strip()[:20] not in pieces[position - 1]]
    assert len(unshared) == 0, (
        "Passages "
        + str(unshared)
        + " of one long section open with text the passage before them does not carry. "
        "Neighbouring passages of a section overlap by up to "
        + str(config.CHUNK_OVERLAP_CHARACTERS)
        + " characters, so a sentence cut at a boundary is still whole in one of them."
    )
    for description, document_text in [
            ("no header line", "MERROWDALE POWER NETWORKS\n1. Overlap check\n" + " ".join(lines)),
            ("two header lines", header + "\n" + header + "\n1. Overlap check\n"
             + " ".join(lines)),
            ("sections numbered from 2", header + "\n2. Overlap check\n" + " ".join(lines))]:
        refusal = None
        try:
            split_documents_into_sections([{"document_name": "Broken-Layout.pdf",
                                            "document_text": document_text}])
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and "Broken-Layout.pdf" in refusal, (
            "A document with "
            + description
            + " was split, giving "
            + str(refusal)
            + ". Every document carries one header line and sections numbered from 1, so a "
            "document laid out otherwise has been read wrongly and is refused, naming it."
        )


def test_index_holds_every_passage_once(knowledge_collection, document_passages):
    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    assert document_passages["problem"] is None, document_passages["problem"]
    collection = knowledge_collection["collection"]
    documents = document_passages["documents"]
    chunks = document_passages["chunks"]
    stored = collection.get(include=["metadatas", "documents"])
    assert collection.count() == len(chunks) == len(stored["ids"]), (
        "The knowledge index holds "
        + str(collection.count())
        + " passages where cutting the documents gives "
        + str(len(chunks))
        + ". More means the index was added to twice and repeats itself in every result; fewer "
        "means passages no search can reach."
    )
    assert len(set(stored["ids"])) == len(stored["ids"]), (
        "Stored passage identifiers repeat, so one passage has overwritten another and its text "
        "can no longer be found."
    )
    names = {document["document_name"] for document in documents}
    cited = [(metadata or {}).get("document_name") for metadata in stored["metadatas"]]
    unnamed = [value for value in cited if value not in names]
    assert len(unnamed) == 0, (
        str(len(unnamed))
        + " stored passages name no document from the folder, for example "
        + str(unnamed[:3])
        + ". The name is what lets an answer cite its source and a section be read from its "
        "own document alone."
    )
    by_document = {}
    for chunk in chunks:
        by_document[chunk["document_name"]] = by_document.get(chunk["document_name"], 0) + 1
    stored_by_document = {name: cited.count(name) for name in names}
    assert stored_by_document == by_document, (
        "Passages stored per document "
        + str(stored_by_document)
        + " differ from passages cut per document "
        + str(by_document)
        + ". A document with passages missing answers less than it says."
    )
    assert sorted(stored["documents"]) == sorted(chunk["chunk_text"] for chunk in chunks), (
        "The texts in the knowledge index are not the passages the documents are cut into, so "
        "a section read from the index is not the section the document states."
    )
    for chunk_id, text, metadata in zip(stored["ids"], stored["documents"], stored["metadatas"],
                                        strict=True):
        twins = [chunk for chunk in chunks if chunk["chunk_id"] == chunk_id]
        assert (len(twins) == 1 and twins[0]["chunk_text"] == text
                and twins[0]["document_name"] == (metadata or {}).get("document_name")), (
            "The stored passage "
            + str(chunk_id)
            + " does not carry the text and document of the passage cut under that identifier."
        )


def test_tools_read_whole_own_sections(knowledge_collection, document_passages):
    from config import config
    from knowledge.vector_index import read_document_section
    from tools.retrieval_tools import (read_connection_approval_guidance,
                                       read_documentation_to_provide,
                                       read_household_notice_guidance, read_reviewer_checks)

    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    assert document_passages["problem"] is None, document_passages["problem"]
    collection = knowledge_collection["collection"]
    unread = []
    for section in document_passages["sections"]:
        read_back = read_document_section(collection, section["document_name"],
                                          section["section_number"])
        if read_back != section["section_text"]:
            unread.append(section["document_name"] + " section "
                          + str(section["section_number"]) + " read as "
                          + repr(str(read_back)[:120]))
    assert len(unread) == 0, (
        str(len(unread))
        + " sections of the "
        + str(len(document_passages["sections"]))
        + " read back from the index differ from the documents, for example "
        + str(unread[:2])
        + ". A section is read back from its own document's passages alone, whole and only "
        "once, exactly as the documents were divided."
    )
    wanted = []
    for technology, document_name in config.CONNECTION_STANDARD_DOCUMENTS.items():
        wanted.append(("the documentation to provide for " + technology, document_name,
                       config.DOCUMENTATION_SECTION_NUMBER, config.DOCUMENTATION_SECTION_TITLE,
                       read_documentation_to_provide(collection, technology)))
        wanted.append(("what the connection approval of " + technology + " says", document_name,
                       config.CONNECTION_APPROVAL_SECTION_NUMBER,
                       config.CONNECTION_APPROVAL_SECTION_TITLE,
                       read_connection_approval_guidance(collection, technology)))
        wanted.append(("what the household notice of " + technology + " says", document_name,
                       config.HOUSEHOLD_NOTICE_SECTION_NUMBER,
                       config.HOUSEHOLD_NOTICE_SECTION_TITLE,
                       read_household_notice_guidance(collection, technology)))
    for sentence_type, document_name in config.SENTENCE_TYPE_GUIDE_DOCUMENTS.items():
        wanted.append(("what the reviewer checks in a sentence of the type " + sentence_type,
                       document_name, config.REVIEWER_CHECKS_SECTION_NUMBER,
                       config.REVIEWER_CHECKS_SECTION_TITLE,
                       read_reviewer_checks(collection, sentence_type)))
    for label, document_name, section_number, section_title, passage in wanted:
        heading = str(section_number) + ". " + section_title + " "
        sections = [section["section_text"] for section in document_passages["sections"]
                    if section["document_name"] == document_name
                    and section["section_text"].startswith(heading)]
        assert len(sections) == 1, (
            document_name
            + " holds "
            + str(len(sections))
            + " sections headed "
            + repr(heading.strip())
            + " where it holds one."
        )
        expected = sections[0][len(heading):]
        assert passage == expected, (
            "Reading "
            + label
            + " gave "
            + repr(str(passage)[:160])
            + " where "
            + document_name
            + "'s section "
            + repr(heading.strip())
            + " reads "
            + repr(expected[:160])
            + ". The tools read that section whole, from that document alone, without its "
            "heading."
        )
    for label, call_name, unknown in [
            ("the documentation of a technology with no standard", "documentation", "WT"),
            ("the connection approval of a technology with no standard", "approval", "CHP"),
            ("the household notice of a technology with no standard", "household", "GSHP-X"),
            ("the reviewer's checks for a sentence type with no guide", "checks", "Rumour"),
            ("a section the index does not hold", "section",
             config.CONNECTION_STANDARD_DOCUMENTS[config.HEAT_PUMP])]:
        refusal = None
        try:
            if call_name == "documentation":
                read_documentation_to_provide(collection, unknown)
            elif call_name == "approval":
                read_connection_approval_guidance(collection, unknown)
            elif call_name == "household":
                read_household_notice_guidance(collection, unknown)
            elif call_name == "checks":
                read_reviewer_checks(collection, unknown)
            else:
                read_document_section(collection, unknown, 40)
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and unknown in refusal, (
            "Asking for "
            + label
            + ", "
            + unknown
            + ", gave "
            + str(refusal)
            + ". It is refused with a ValueError naming it, never answered from another "
            "document or section."
        )


def test_chat_search_skips_casebook_nearest_first(knowledge_collection):
    from config import config
    from knowledge.vector_index import search_documents

    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    collection = knowledge_collection["collection"]
    casebook_passages = []
    for document_name in config.CHAT_EXCLUDED_DOCUMENTS:
        casebook_passages.extend(collection.get(where={"document_name": document_name})["ids"])
    assert len(casebook_passages) > 0, (
        "The index holds no passage of "
        + str(config.CHAT_EXCLUDED_DOCUMENTS)
        + ", which is indexed with the other documents and left out of the chat's search only."
    )
    stored = collection.get(include=["documents", "metadatas"])
    longest = {}
    for text, metadata in zip(stored["documents"], stored["metadatas"], strict=True):
        document_name = (metadata or {}).get("document_name")
        if (document_name not in config.CHAT_EXCLUDED_DOCUMENTS
                and len(text) > len(longest.get(document_name, ""))):
            longest[document_name] = text
    questions = [(text, True) for text in longest.values()]
    questions.append(("What did Network Planning decide when a past application was referred "
                      "to it?", False))
    questions.append(("LCT0041327 was sent from SiteWren for a heat pump replacing a failed "
                      "boiler.", False))
    results = search_documents(collection, [pair[0] for pair in questions])
    assert isinstance(results, list) and len(results) == len(questions), (
        "search_documents answered "
        + str(results)[:200]
        + " for "
        + str(len(questions))
        + " questions, where it answers one list of passages a question, in order."
    )
    for (question, stored_passage), passages in zip(questions, results, strict=True):
        assert (isinstance(passages, list) and len(passages) == config.CHAT_RETRIEVAL_RESULT_COUNT
                and all(isinstance(passage, dict)
                        and {"chunk_id", "chunk_text", "document_name", "distance"} <= set(passage)
                        for passage in passages)), (
            "The search for "
            + repr(question[:80])
            + " returned "
            + str(passages)[:200]
            + " where it returns "
            + str(config.CHAT_RETRIEVAL_RESULT_COUNT)
            + " passages, each with its identifier, text, document and distance."
        )
        distances = [float(passage["distance"]) for passage in passages]
        found = [passage["document_name"] for passage in passages]
        assert distances == sorted(distances) and all(-0.0001 <= value <= 2.0001
                                                      for value in distances), (
            "The search for "
            + repr(question[:80])
            + " returned distances "
            + str([round(value, 3) for value in distances])
            + ". Passages come nearest first, at cosine distances between 0 and 2."
        )
        excluded = [name for name in found if name in config.CHAT_EXCLUDED_DOCUMENTS]
        assert len(excluded) == 0, (
            "The search for "
            + repr(question[:80])
            + " returned passages of "
            + str(excluded)
            + ". The casebook records past reviews of named applications and is never quoted by "
            "the chat, however close its passages are."
        )
        if stored_passage:
            assert passages[0]["chunk_text"] == question, (
                "Searching with the words of the stored passage "
                + repr(question[:80])
                + " returned "
                + repr(str(passages[0]["chunk_text"])[:80])
                + " first. No passage is nearer to a passage's own words than the passage "
                "itself, so it comes back first."
            )


def test_each_technology_read_from_own_standard(invented_standards):
    from config import config

    assert invented_standards["problem"] is None, invented_standards["problem"]
    standards = invented_standards["standards"]
    assert sorted(standards) == sorted(config.CONNECTION_STANDARD_DOCUMENTS), (
        "Standards were read for "
        + str(sorted(standards))
        + " where every one of the five technologies has its own."
    )
    for technology, requirements in invented_standards["invented"].items():
        standard = standards[technology]
        document_name = config.CONNECTION_STANDARD_DOCUMENTS[technology]
        assert (standard.get("technology") == technology
                and standard.get("document_name") == document_name), (
            "The standard read for "
            + technology
            + " names the technology "
            + repr(standard.get("technology"))
            + " and the document "
            + repr(standard.get("document_name"))
            + ". A standard is read from the document configured for its own technology and "
            "names both."
        )
        read = [(requirement.get("requirement_name"), requirement.get("requirement_kind"))
                for requirement in standard.get("requirements", [])]
        stated = [(requirement[1], requirement[0]) for requirement in requirements]
        assert read == stated, (
            "A standard written for the checks under the name "
            + document_name
            + " was read as the requirements "
            + str(read)
            + " where its section 4 states "
            + str(stated)
            + ", in that order. Each numbered requirement is read whole, named as the standard "
            "names it, from the standard of its own technology alone: the EV charge point and "
            "heat pump standards read alike, and neither lends the other a requirement."
        )


def test_requirements_keep_sentence_and_figures(invented_standards):
    from decimal import Decimal

    from config import config

    assert invented_standards["problem"] is None, invented_standards["problem"]
    amps, ohms = config.MAXIMUM_AMPS_RULE, config.MAXIMUM_OHMS_RULE
    power_factor = config.MINIMUM_POWER_FACTOR_RULE
    stated = {
        config.EV_CHARGE_POINT: [{}, {amps: Decimal("28")}, {ohms: Decimal("165")}, {}],
        config.HEAT_PUMP: [{}, {}, {amps: Decimal("21.5")}, {power_factor: Decimal("0.92")},
                           {ohms: Decimal("140")}],
        config.VEHICLE_TO_GRID: [{}, {}, {amps: Decimal("26")}, {}, {amps: Decimal("13")}],
        config.SOLAR_PV: [{}, {}, {amps: Decimal("11.5")}, {power_factor: Decimal("0.9")}],
        config.BATTERY_STORAGE: [{}, {amps: Decimal("17")}, {}, {}, {amps: Decimal("14")},
                                 {ohms: Decimal("95.5")}],
    }
    for technology, figures in stated.items():
        document_name = config.CONNECTION_STANDARD_DOCUMENTS[technology]
        requirements = invented_standards["standards"][technology].get("requirements", [])
        written = invented_standards["invented"][technology]
        assert len(requirements) == len(figures), (
            document_name
            + " was read as "
            + str(len(requirements))
            + " requirements where it states "
            + str(len(figures))
            + "."
        )
        for requirement, wanted, stated_requirement in zip(requirements, figures, written,
                                                           strict=True):
            sentence = requirement.get("requirement_sentence")
            assert sentence == stated_requirement[2], (
                "The requirement "
                + str(requirement.get("requirement_name"))
                + " of "
                + document_name
                + " keeps the sentence "
                + repr(str(sentence)[:160])
                + " where the standard states "
                + repr(stated_requirement[2])
                + ". A requirement keeps the sentence it was read from, word for word, so the "
                "review and the rework request quote the standard itself."
            )
            rules = requirement.get("rules", {})
            exact = isinstance(rules, dict) and all(isinstance(figure, Decimal)
                                                    for figure in rules.values())
            assert rules == wanted and exact, (
                "The requirement "
                + str(requirement.get("requirement_name"))
                + " of "
                + document_name
                + " was read with the figures "
                + repr(rules)
                + " where its sentence states "
                + repr(wanted)
                + ". Each requirement keeps its own figures, read from its own standard as exact "
                "decimals, and no others; none of these figures is Merrowdale's, so none can "
                "come from anywhere but the standard read."
            )


def test_twin_currents_read_from_own_documents(invented_standards):
    from decimal import Decimal

    from config import config
    from knowledge.standard_reading import read_connection_standard

    assert invented_standards["problem"] is None, invented_standards["problem"]
    documents = invented_standards["documents"]
    reordered = list(reversed(documents))
    for technology in config.CONNECTION_STANDARD_DOCUMENTS:
        again = read_connection_standard(technology, reordered)
        assert again == invented_standards["standards"][technology], (
            "The standard of "
            + technology
            + " reads differently when the documents are handed over in another order. A "
            "standard is found by its document's name, wherever it stands among the documents."
        )
    heat_pump_document = config.CONNECTION_STANDARD_DOCUMENTS[config.HEAT_PUMP]
    amended = []
    for document in documents:
        document_text = document["document_text"]
        if document["document_name"] == heat_pump_document:
            document_text = document_text.replace("is at most 21.5 amps per phase",
                                                  "is at most 23 amps per phase")
        amended.append({"document_name": document["document_name"],
                        "document_text": document_text})
    amended_currents = {}
    for technology in [config.EV_CHARGE_POINT, config.HEAT_PUMP]:
        amended_currents[technology] = [
            requirement.get("rules", {}).get(config.MAXIMUM_AMPS_RULE)
            for requirement in read_connection_standard(technology, amended).get(
                "requirements", [])
            if requirement.get("requirement_kind") == config.RATED_CURRENT_REQUIREMENT]
    assert amended_currents == {config.EV_CHARGE_POINT: [Decimal("28")],
                                config.HEAT_PUMP: [Decimal("23")]}, (
        "With only the heat pump standard's rated current changed to 23 amps, the two twins "
        "read "
        + str(amended_currents)
        + ". Changing one standard changes that technology's limit and no other's, because each "
        "is read from its own document."
    )


def test_standards_read_alike_however_laid_out(invented_standards):
    from config import config
    from knowledge.standard_reading import read_connection_standard

    assert invented_standards["problem"] is None, invented_standards["problem"]
    standard_documents = set(config.CONNECTION_STANDARD_DOCUMENTS.values())
    relaid = []
    for document in invented_standards["documents"]:
        document_text = document["document_text"]
        if document["document_name"] in standard_documents:
            document_text = (document_text.replace(" is at most ", " is at\nmost  ")
                             .replace(" amps per phase", "\namps per   phase")
                             .replace(", and on a TT earth", ",\n and on a TT earth")
                             .replace(" is listed in the ", " is listed\nin the "))
        relaid.append({"document_name": document["document_name"],
                       "document_text": document_text})
    differing = []
    for technology in config.CONNECTION_STANDARD_DOCUMENTS:
        try:
            again = read_connection_standard(technology, relaid)
        except Exception as error:
            again = type(error).__name__ + ": " + str(error)
        if again != invented_standards["standards"][technology]:
            differing.append(technology + " read as " + repr(again)[:200])
    assert len(differing) == 0, (
        "With each standard's lines broken in other places and runs of spaces between its "
        "words, as text taken from a PDF can come, the standards read differently: "
        + str(differing[:2])
        + ". A standard is read from its sections, in which every run of spaces and line breaks "
        "is one space, so the same standard reads the same however its lines fall."
    )


def test_unknown_technology_refused_reading_repeatable(invented_standards):
    from config import config
    from knowledge.standard_reading import read_connection_standard

    assert invented_standards["problem"] is None, invented_standards["problem"]
    documents = invented_standards["documents"]
    refusal = None
    try:
        read_connection_standard("WT", documents)
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    assert refusal is not None and "WT" in refusal, (
        "Reading the standard of a technology with none configured gave "
        + str(refusal)
        + ". A technology with no standard has no requirements to screen against, and is "
        "refused with a ValueError naming it."
    )
    for technology in config.CONNECTION_STANDARD_DOCUMENTS:
        again = read_connection_standard(technology, documents)
        assert again == invented_standards["standards"][technology], (
            "The standard of "
            + technology
            + " reads differently the second time. A standard is read from its sections, the "
            "same way every time."
        )
    solar_standard = config.CONNECTION_STANDARD_DOCUMENTS[config.SOLAR_PV]
    refusal = None
    try:
        read_connection_standard(config.SOLAR_PV,
                                 [document for document in documents
                                  if document["document_name"] != solar_standard])
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    assert refusal is not None and solar_standard in refusal, (
        "Reading the solar PV standard from documents that do not include it gave "
        + str(refusal)
        + ". A standard missing from the documents read leaves its technology with nothing to "
        "screen against, so it is refused with a ValueError naming the document."
    )
    registered = "Registered equipment. The equipment is listed in the type-test register."
    current = "Rated current. The rated current is at most 30 amps per phase."
    isolation = "Isolation. An isolator that locks off is fitted on the inverter's mains side."
    broken = {
        config.EV_CHARGE_POINT: (config.HEAT_PUMP, ["1 " + registered, "2 " + current]),
        config.HEAT_PUMP: (config.HEAT_PUMP, ["1 " + registered, "2 " + current,
                                              "3 " + current.replace("30", "25")]),
        config.VEHICLE_TO_GRID: (config.VEHICLE_TO_GRID, ["1 " + registered, "2 " + current,
                                                          "4 " + isolation]),
        config.SOLAR_PV: (config.SOLAR_PV, ["1 " + registered, "2 Noise level. The inverter is "
                                                               "at most 40 decibels at the "
                                                               "boundary."]),
    }
    broken_documents = []
    for technology, (stated_code, requirement_lines) in broken.items():
        broken_documents.append({
            "document_name": config.CONNECTION_STANDARD_DOCUMENTS[technology],
            "document_text": (
                "MERROWDALE POWER NETWORKS\nDocument Standard 98 | Issue 1 | Owner: Engineering "
                "Standards | Review cycle: every twelve months\n1. Purpose and scope\nThis "
                "standard is laid out wrongly on purpose.\n2. What this technology covers\nThe "
                "technology's code is "
                + stated_code
                + ".\n3. Who applies\nThe installer who fits the equipment.\n4. Requirements\n"
                "Every requirement below must be Met, in this order:\n"
                + "\n".join(requirement_lines)
                + "\nA requirement is never met by a sentence in a statement, and is never "
                "waived.\n"),
        })
    broken_documents.append({
        "document_name": config.CONNECTION_STANDARD_DOCUMENTS[config.BATTERY_STORAGE],
        "document_text": (
            "MERROWDALE POWER NETWORKS\nDocument Standard 98 | Issue 1 | Owner: Engineering "
            "Standards | Review cycle: every twelve months\n1. Purpose and scope\nThis standard "
            "is laid out wrongly on purpose.\n2. What this technology covers\nThe technology's "
            "code is BESS.\n3. Who applies\nThe installer who fits the equipment.\n"),
    })
    meanings = {
        config.EV_CHARGE_POINT: "states another technology's code",
        config.HEAT_PUMP: "states the rated current twice",
        config.VEHICLE_TO_GRID: "numbers its requirements 1, 2 and 4",
        config.SOLAR_PV: "states a requirement the screening cannot read",
        config.BATTERY_STORAGE: "has no requirements section",
    }
    for technology, meaning in meanings.items():
        document_name = config.CONNECTION_STANDARD_DOCUMENTS[technology]
        refusal = None
        try:
            read_connection_standard(technology, broken_documents)
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and document_name in refusal, (
            "A standard that "
            + meaning
            + " was read, giving "
            + str(refusal)
            + ". A standard that cannot be read as the Connection Rules lay it out would screen "
            "applications against requirements nobody wrote, so it is refused with a ValueError "
            "naming the document."
        )


def test_requirements_met_at_boundary(live_files):
    from decimal import Decimal

    from config import config
    from tools.application_data_tools import fetch_application
    from tools.screening_tools import screen_application

    assert live_files["problem"] is None, live_files["problem"]
    applications = live_files["applications"]
    templates = {}
    for technology in [config.HEAT_PUMP, config.BATTERY_STORAGE]:
        first_reference = applications.loc[applications[config.TECHNOLOGY_COLUMN] == technology,
                                           config.APPLICATION_REFERENCE_COLUMN].iloc[0]
        templates[technology] = fetch_application(applications, first_reference)
    invented = {
        config.HEAT_PUMP: [
            (config.RATED_CURRENT_REQUIREMENT, "Rated current",
             {config.MAXIMUM_AMPS_RULE: Decimal("27.3")}),
            (config.POWER_FACTOR_REQUIREMENT, "Power factor",
             {config.MINIMUM_POWER_FACTOR_RULE: Decimal("0.97")}),
            (config.EARTHING_PROTECTION_REQUIREMENT, "Earthing protection",
             {config.MAXIMUM_OHMS_RULE: Decimal("137.3")})],
        config.BATTERY_STORAGE: [
            (config.CHARGING_CURRENT_REQUIREMENT, "Charging current",
             {config.MAXIMUM_AMPS_RULE: Decimal("18.3")}),
            (config.EXPORT_LIMITATION_REQUIREMENT, "Export limitation",
             {config.MAXIMUM_AMPS_RULE: Decimal("12.6")})],
    }
    rated, charging = config.RATED_CURRENT_COLUMN, config.CHARGING_CURRENT_COLUMN
    supply, equipment = config.SUPPLY_PHASES_COLUMN, config.EQUIPMENT_PHASES_COLUMN
    earthing, electrode = config.EARTHING_ARRANGEMENT_COLUMN, config.EARTH_ELECTRODE_COLUMN
    fitted = config.EXPORT_LIMITATION_FITTED_COLUMN
    certified = config.EXPORT_SCHEME_CERTIFIED_COLUMN
    export_limit = config.EXPORT_LIMIT_COLUMN
    met, not_met = config.REQUIREMENT_MET, config.REQUIREMENT_NOT_MET
    heat_pump, battery = config.HEAT_PUMP, config.BATTERY_STORAGE
    single = {supply: 1, equipment: 1}
    three = {supply: 3, equipment: 3}
    scheme = {fitted: config.BOOLEAN_YES, certified: config.BOOLEAN_YES}
    cases = [
        (heat_pump, "Rated current", dict(single, **{rated: 27.3}), met),
        (heat_pump, "Rated current", dict(single, **{rated: 27.4}), not_met),
        (heat_pump, "Rated current", dict(three, **{rated: 27.3}), met),
        (heat_pump, "Rated current", dict(three, **{rated: 27.4}), not_met),
        (heat_pump, "Power factor", {config.POWER_FACTOR_COLUMN: 0.97}, met),
        (heat_pump, "Power factor", {config.POWER_FACTOR_COLUMN: 0.9699}, not_met),
        (heat_pump, "Earthing protection", {earthing: config.EARTHING_TT, electrode: 137.3}, met),
        (heat_pump, "Earthing protection", {earthing: config.EARTHING_TT, electrode: 137.4},
         not_met),
        (battery, "Charging current", dict(single, **{charging: 18.3}), met),
        (battery, "Charging current", dict(single, **{charging: 18.4}), not_met),
        (battery, "Export limitation", dict(single, **scheme, **{export_limit: 2.898}), met),
        (battery, "Export limitation", dict(single, **scheme, **{export_limit: 2.921}), not_met),
        (battery, "Export limitation", dict(three, **scheme, **{export_limit: 8.694}), met),
        (battery, "Export limitation", dict(three, **scheme, **{export_limit: 8.763}), not_met),
    ]
    wrong = []
    for technology, requirement_name, answers, expected in cases:
        standard = {"technology": technology,
                    "document_name": "Invented-" + technology + "-Standard.pdf",
                    "requirements": [{"requirement_kind": kind, "requirement_name": name,
                                      "requirement_sentence": "An invented requirement: " + name
                                                              + ".",
                                      "rules": rules}
                                     for kind, name, rules in invented[technology]]}
        application = dict(templates[technology])
        application[config.APPLICATION_REFERENCE_COLUMN] = "LCT0950101"
        application.update(answers)
        screening = screen_application(application, standard)
        assert isinstance(screening, dict) and isinstance(screening.get("requirements"), list), (
            "screen_application returned "
            + str(screening)[:200]
            + ", where it returns the screening with one result for each requirement."
        )
        results = {requirement.get("requirement_name"): requirement.get("result")
                   for requirement in screening["requirements"] if isinstance(requirement, dict)}
        if results.get(requirement_name) != expected:
            wrong.append(requirement_name + " with " + str(answers) + " is "
                         + str(results.get(requirement_name)) + ", not " + expected)
    assert len(wrong) == 0, (
        "Under standards whose figures no Merrowdale standard uses, the screening misplaces "
        "boundaries: "
        + "; ".join(wrong[:5])
        + ". A maximum is met at the figure itself and a minimum likewise, every current in "
        "amps per phase and every figure compared as an exact decimal - a rated current of "
        "27.3 against a limit of 27.3, or a power factor of 0.97 against 0.97, is met - and an "
        "export limit in kilowatts is made amps per phase on the equipment's phases first; "
        "every figure comes from the standard handed over, never from a figure written into "
        "the code."
    )


def test_earthing_protection_follows_arrangement(live_files):
    from decimal import Decimal

    from config import config
    from tools.application_data_tools import fetch_application
    from tools.screening_tools import screen_application

    assert live_files["problem"] is None, live_files["problem"]
    applications = live_files["applications"]
    first_reference = applications.loc[
        applications[config.TECHNOLOGY_COLUMN] == config.EV_CHARGE_POINT,
        config.APPLICATION_REFERENCE_COLUMN].iloc[0]
    template = fetch_application(applications, first_reference)
    standard = {"technology": config.EV_CHARGE_POINT,
                "document_name": "Invented-Earthing-Standard.pdf",
                "requirements": [
                    {"requirement_kind": config.EARTHING_PROTECTION_REQUIREMENT,
                     "requirement_name": "Earthing protection",
                     "requirement_sentence": ("On a PME earth the equipment has open-PEN "
                                              "protection, and on a TT earth its earth electrode "
                                              "is at most 137.3 ohms."),
                     "rules": {config.MAXIMUM_OHMS_RULE: Decimal("137.3")}},
                    {"requirement_kind": config.REGISTERED_EQUIPMENT_REQUIREMENT,
                     "requirement_name": "Registered equipment",
                     "requirement_sentence": "The equipment is listed in the approved register.",
                     "rules": {}}]}
    earthing, open_pen = config.EARTHING_ARRANGEMENT_COLUMN, config.OPEN_PEN_PROTECTION_COLUMN
    electrode = config.EARTH_ELECTRODE_COLUMN
    tn_c_s, tn_s, tt_earth = config.EARTHING_TN_C_S, config.EARTHING_TN_S, config.EARTHING_TT
    answered_yes, answered_no = config.BOOLEAN_YES, config.BOOLEAN_NO
    met, not_met = config.REQUIREMENT_MET, config.REQUIREMENT_NOT_MET
    undocumented = config.REQUIREMENT_NOT_DOCUMENTED
    cases = [
        ({earthing: tn_s, open_pen: None, electrode: None}, met, []),
        ({earthing: tn_s, open_pen: answered_no, electrode: 500.0}, met, []),
        ({earthing: tn_c_s, open_pen: answered_yes, electrode: None}, met, []),
        ({earthing: tn_c_s, open_pen: answered_no, electrode: 10.0}, not_met, []),
        ({earthing: tn_c_s, open_pen: None, electrode: 10.0}, undocumented, [open_pen]),
        ({earthing: tt_earth, open_pen: None, electrode: 137.3}, met, []),
        ({earthing: tt_earth, open_pen: answered_yes, electrode: 137.4}, not_met, []),
        ({earthing: tt_earth, open_pen: answered_yes, electrode: None}, undocumented, [electrode]),
        ({earthing: None, open_pen: answered_yes, electrode: 50.0}, undocumented, [earthing]),
        ({earthing: None, open_pen: None, electrode: None}, undocumented, [earthing]),
    ]
    wrong = []
    for answers, expected, expected_blanks in cases:
        application = dict(template)
        application[config.APPLICATION_REFERENCE_COLUMN] = "LCT0950102"
        application.update(answers)
        screening = screen_application(application, standard)
        assert (isinstance(screening, dict) and isinstance(screening.get("requirements"), list)
                and len(screening["requirements"]) >= 1
                and isinstance(screening["requirements"][0], dict)), (
            "screen_application returned "
            + str(screening)[:200]
            + ", where it returns the screening with one result for each of the standard's "
            "requirements, in the standard's order."
        )
        requirement = screening["requirements"][0]
        if (requirement.get("result") != expected
                or list(requirement.get("blank_answers", [])) != expected_blanks):
            wrong.append(str(answers) + " gives " + str(requirement.get("result")) + " asking for "
                         + str(requirement.get("blank_answers")) + ", not " + expected
                         + " asking for " + str(expected_blanks))
    assert len(wrong) == 0, (
        "Earthing protection is screened wrongly: "
        + "; ".join(wrong[:4])
        + ". It is a follow-on question decided by the earthing arrangement: Met on TN-S "
        "whatever else the form says; on TN-C-S Met with open-PEN protection, Not met without "
        "it and Not documented when that answer is blank, asking for it; on TT Met with an "
        "electrode within the figure, Not met above it and Not documented when the resistance "
        "is blank, asking for it alone; and Not documented, asking for the arrangement alone, "
        "when the arrangement is blank."
    )
    for answer, expected, expected_blanks in [(answered_yes, met, []), (answered_no, not_met, []),
                                              (None, undocumented,
                                               [config.REGISTER_LISTED_COLUMN])]:
        application = dict(template)
        application[config.APPLICATION_REFERENCE_COLUMN] = "LCT0950103"
        application[config.REGISTER_LISTED_COLUMN] = answer
        screening = screen_application(application, standard)
        assert (isinstance(screening, dict) and isinstance(screening.get("requirements"), list)
                and len(screening["requirements"]) >= 2
                and isinstance(screening["requirements"][1], dict)), (
            "screen_application returned "
            + str(screening)[:200]
            + ", where it returns the screening with one result for each of the standard's "
            "requirements, in the standard's order."
        )
        requirement = screening["requirements"][1]
        assert (requirement.get("result") == expected
                and list(requirement.get("blank_answers", [])) == expected_blanks), (
            "Equipment answered "
            + repr(answer)
            + " for whether it is listed in the register is screened "
            + str(requirement.get("result"))
            + " asking for "
            + str(requirement.get("blank_answers"))
            + ". A yes-or-no requirement is Met on Yes, Not met on No and Not documented when "
            "the answer is blank, naming the answer it needs."
        )


def test_export_limitation_asked_only_when_fitted(live_files):
    from decimal import Decimal

    from config import config
    from tools.application_data_tools import fetch_application
    from tools.notice_tools import build_approval_details
    from tools.screening_tools import screen_application

    assert live_files["problem"] is None, live_files["problem"]
    applications = live_files["applications"]
    templates = {}
    for technology in [config.SOLAR_PV, config.HEAT_PUMP]:
        first_reference = applications.loc[applications[config.TECHNOLOGY_COLUMN] == technology,
                                           config.APPLICATION_REFERENCE_COLUMN].iloc[0]
        templates[technology] = fetch_application(applications, first_reference)
    export_requirement = {"requirement_kind": config.EXPORT_LIMITATION_REQUIREMENT,
                          "requirement_name": "Export limitation",
                          "requirement_sentence": ("An export limitation scheme, where one is "
                                                   "fitted, is certified and limits export to at "
                                                   "most 12.6 amps per phase."),
                          "rules": {config.MAXIMUM_AMPS_RULE: Decimal("12.6")}}
    solar_standard = {"technology": config.SOLAR_PV,
                      "document_name": "Invented-Solar-Standard.pdf",
                      "requirements": [export_requirement]}
    fitted = config.EXPORT_LIMITATION_FITTED_COLUMN
    certified = config.EXPORT_SCHEME_CERTIFIED_COLUMN
    export_limit, equipment = config.EXPORT_LIMIT_COLUMN, config.EQUIPMENT_PHASES_COLUMN
    answered_yes, answered_no = config.BOOLEAN_YES, config.BOOLEAN_NO
    met, not_met = config.REQUIREMENT_MET, config.REQUIREMENT_NOT_MET
    undocumented = config.REQUIREMENT_NOT_DOCUMENTED
    cases = [
        (answered_no, None, None, 1, met, []), (answered_no, answered_no, 9.9, 1, met, []),
        (answered_yes, answered_yes, 2.898, 1, met, []),
        (answered_yes, answered_yes, 8.694, 3, met, []),
        (answered_yes, answered_no, None, 1, not_met, []),
        (answered_yes, None, 5.0, 1, not_met, []),
        (answered_yes, answered_yes, 2.921, None, undocumented, [equipment]),
        (answered_yes, None, 2.0, 1, undocumented, [certified]),
        (answered_yes, answered_yes, None, 1, undocumented, [export_limit]),
        (answered_yes, None, None, 1, undocumented, [certified, export_limit]),
        (None, answered_yes, 2.0, 1, undocumented, [fitted]),
        (None, None, None, None, undocumented, [fitted]),
    ]
    wrong = []
    for written_fitted, written_certified, written_limit, phases, expected, expected_blanks in (
            cases):
        application = dict(templates[config.SOLAR_PV])
        application[config.APPLICATION_REFERENCE_COLUMN] = "LCT0950105"
        application.update({fitted: written_fitted, certified: written_certified,
                            export_limit: written_limit, equipment: phases,
                            config.SUPPLY_PHASES_COLUMN: 3})
        screening = screen_application(application, solar_standard)
        assert (isinstance(screening, dict) and isinstance(screening.get("requirements"), list)
                and len(screening["requirements"]) >= 1
                and isinstance(screening["requirements"][0], dict)), (
            "screen_application returned "
            + str(screening)[:200]
            + ", where it returns the screening with one result for each of the standard's "
            "requirements, in the standard's order."
        )
        requirement = screening["requirements"][0]
        if (requirement.get("result") != expected
                or list(requirement.get("blank_answers", [])) != expected_blanks):
            wrong.append("fitted " + str(written_fitted) + ", certified " + str(written_certified)
                         + ", a limit of " + str(written_limit) + " kW on "
                         + str(phases) + " phases gives " + str(requirement.get("result"))
                         + " asking for " + str(requirement.get("blank_answers")) + ", not "
                         + expected + " asking for " + str(expected_blanks))
    assert len(wrong) == 0, (
        "The export limitation requirement is screened wrongly: "
        + "; ".join(wrong[:4])
        + ". It is a follow-on question asked only where a scheme is fitted: with no scheme it "
        "is Met; a scheme known to be uncertified, or whose limit made amps per phase is above "
        "the figure, is Not met whatever else is blank; otherwise a blank certification or "
        "limit is Not documented, asking for each blank answer, and a limit that cannot be made "
        "amps per phase asks for the equipment's phases; and a blank answer to whether a scheme "
        "is fitted is Not documented, asking for that answer alone."
    )
    heat_pump_standard = {"technology": config.HEAT_PUMP,
                          "document_name": "Invented-Heat-Pump-Standard.pdf",
                          "requirements": [
                              {"requirement_kind": config.SINGLE_CONTROLLER_REQUIREMENT,
                               "requirement_name": "Single controller",
                               "requirement_sentence": ("The equipment and every back-up or "
                                                        "boost heater are run by a single "
                                                        "controller."),
                               "rules": {}}]}
    written = {config.APPLICATION_REFERENCE_COLUMN: "LCT0950108",
               equipment: 1, config.SUPPLY_PHASES_COLUMN: 1,
               config.EARTHING_ARRANGEMENT_COLUMN: config.EARTHING_TN_S,
               config.RATED_CURRENT_COLUMN: 16.0, config.INVERTER_CAPACITY_COLUMN: 3.68,
               fitted: answered_yes, certified: answered_yes, export_limit: 3.0}
    for technology, standard, conditions in [
            (config.SOLAR_PV, solar_standard,
             {config.INVERTER_CAPACITY_COLUMN: 3.68, export_limit: 3.0}),
            (config.HEAT_PUMP, heat_pump_standard,
             {config.RATED_CURRENT_COLUMN: 16.0, config.LOAD_LIMITER_SETTING_COLUMN: 40.0})]:
        application = dict(templates[technology], **written)
        application[config.LOAD_LIMITER_FITTED_COLUMN] = answered_no
        application[config.LOAD_LIMITER_SETTING_COLUMN] = None
        if technology == config.HEAT_PUMP:
            application[config.LOAD_LIMITER_FITTED_COLUMN] = answered_yes
            application[config.LOAD_LIMITER_SETTING_COLUMN] = 40.0
        screening = screen_application(application, standard)
        details = build_approval_details(application, screening)
        expected_details = {config.APPLICATION_REFERENCE_COLUMN: "LCT0950108",
                            "standard_document": standard["document_name"],
                            config.TECHNOLOGY_COLUMN: technology, equipment: 1,
                            config.EARTHING_ARRANGEMENT_COLUMN: config.EARTHING_TN_S,
                            **conditions}
        assert details == expected_details, (
            "The connection approval details of a "
            + config.TECHNOLOGY_NAMES[technology]
            + " application are "
            + str(details)
            + " where they are "
            + str(expected_details)
            + ". The connection approval gives the reference, the standard, the technology, the "
            "equipment's phases, the earthing arrangement and the technology's rating as the "
            "form gives them, the export limit only for a technology that exports and a load "
            "limiting device's setting only where one is set, because the connection depends on "
            "them."
        )


def test_network_triggers_follow_rules(live_files):
    from config import config
    from tools.application_data_tools import fetch_application
    from tools.screening_tools import find_network_triggers

    assert live_files["problem"] is None, live_files["problem"]
    applications = live_files["applications"]
    templates = {}
    for technology in [config.HEAT_PUMP, config.SOLAR_PV]:
        first_reference = applications.loc[applications[config.TECHNOLOGY_COLUMN] == technology,
                                           config.APPLICATION_REFERENCE_COLUMN].iloc[0]
        templates[technology] = fetch_application(applications, first_reference)
    looped, cut_out = config.LOOPED_SERVICE_COLUMN, config.CUT_OUT_RATING_COLUMN
    demand, limiter = config.MAXIMUM_DEMAND_COLUMN, config.LOAD_LIMITER_FITTED_COLUMN
    setting, phases = config.LOAD_LIMITER_SETTING_COLUMN, config.EQUIPMENT_PHASES_COLUMN
    inverter, existing = config.INVERTER_CAPACITY_COLUMN, config.EXISTING_GENERATION_COLUMN
    fitted = config.EXPORT_LIMITATION_FITTED_COLUMN
    certified = config.EXPORT_SCHEME_CERTIFIED_COLUMN
    export_limit = config.EXPORT_LIMIT_COLUMN
    answered_yes, answered_no = config.BOOLEAN_YES, config.BOOLEAN_NO
    looped_trigger, capacity = config.LOOPED_SERVICE_TRIGGER, config.SUPPLY_CAPACITY_TRIGGER
    generation = config.GENERATION_TRIGGER
    heat_pump, solar = config.HEAT_PUMP, config.SOLAR_PV
    capped = {fitted: answered_yes, certified: answered_yes, export_limit: 3.68}
    cases = [
        (heat_pump, {}, []), (heat_pump, {looped: answered_yes}, [looped_trigger]),
        (heat_pump, {looped: None}, [looped_trigger]),
        (heat_pump, {demand: 60.0}, []), (heat_pump, {demand: 60.1}, [capacity]),
        (heat_pump, {demand: 44.9, cut_out: 45}, []),
        (heat_pump, {demand: 45.0, cut_out: 45}, [capacity]),
        (heat_pump, {demand: None}, [capacity]), (heat_pump, {cut_out: None}, [capacity]),
        (heat_pump, {demand: 90.0, limiter: answered_yes, setting: 55.0}, []),
        (heat_pump, {demand: 40.0, limiter: answered_yes, setting: 61.0}, [capacity]),
        (heat_pump, {demand: 40.0, limiter: answered_yes, setting: None}, [capacity]),
        (heat_pump, {demand: 90.0, limiter: answered_no, setting: 30.0}, [capacity]),
        (heat_pump, {demand: 50.0, limiter: None, setting: 70.0}, [capacity]),
        (heat_pump, {demand: 50.0, limiter: None, setting: 40.0}, []),
        (heat_pump, {demand: None, limiter: None, setting: 40.0}, [capacity]),
        (heat_pump, {inverter: 9.0, existing: 4.0}, []),
        (heat_pump, {looped: None, demand: None}, [looped_trigger, capacity]),
        (solar, {}, []), (solar, {inverter: 3.71}, [generation]),
        (solar, {inverter: 2.0, existing: 1.7}, [generation]),
        (solar, {inverter: 5.0, **capped}, []), (solar, {inverter: 7.36, **capped}, []),
        (solar, {inverter: 7.4, **capped}, [generation]),
        (solar, {inverter: 5.0, **dict(capped, **{certified: answered_no})}, [generation]),
        (solar, {inverter: 5.0, **dict(capped, **{certified: None})}, [generation]),
        (solar, {inverter: 5.0, **dict(capped, **{export_limit: 4.0})}, [generation]),
        (solar, {inverter: 5.0, **dict(capped, **{fitted: answered_no})}, [generation]),
        (solar, {inverter: 11.04, phases: 3}, []), (solar, {inverter: 11.1, phases: 3},
                                                     [generation]),
        (solar, {inverter: None}, [generation]), (solar, {existing: None}, [generation]),
        (solar, {phases: None}, [generation]),
        (solar, {looped: answered_yes, demand: 61.0, inverter: 3.71},
         [looped_trigger, capacity, generation]),
    ]
    base = {looped: answered_no, cut_out: 100, demand: 50.0, limiter: answered_no, setting: None,
            phases: 1, config.SUPPLY_PHASES_COLUMN: 3, inverter: 3.68, existing: 0.0,
            fitted: answered_no, certified: None, export_limit: None}
    wrong = []
    for technology, answers, expected in cases:
        application = dict(templates[technology])
        application[config.APPLICATION_REFERENCE_COLUMN] = "LCT0950106"
        application.update(base)
        application.update(answers)
        triggers = find_network_triggers(application)
        if not isinstance(triggers, list) or triggers != expected:
            wrong.append("a " + config.TECHNOLOGY_NAMES[technology] + " with " + str(answers)
                         + " meets " + str(triggers) + ", not " + str(expected))
    assert len(wrong) == 0, (
        "The network triggers are found wrongly, on a form whose service is not looped, with "
        "a demand of 50 amps per phase on a 100-amp cut-out, no load limiting device, and for "
        "generation a 3.68 kW single-phase inverter with no existing generation and no scheme: "
        + "; ".join(wrong[:4])
        + ". A looped service triggers; so does a demand at or above the cut-out rating or "
        "above 60 amps per phase, a load limiting device's setting counting in place of the "
        "declared demand where one is fitted and the larger of the two where that answer is "
        "blank; generation triggers above 16 amps per phase on the equipment's phases, the "
        "existing generation added in, unless a certified scheme caps the export at 16 amps "
        "per phase on an installation of at most 32, the fast track; and a blank answer counts "
        "on the network's side. The triggers are listed in that order."
    )


def test_assessment_decisions_and_worked_example(live_files, tmp_path):
    from decimal import Decimal

    import pandas as pd

    from config import config
    from data_pipeline.application_cleaning import clean_application_records
    from data_pipeline.dataset_loader import load_raw_dataset
    from tools.application_data_tools import fetch_application
    from tools.screening_tools import screen_application

    assert live_files["problem"] is None, live_files["problem"]
    applications = live_files["applications"]
    templates = {}
    for technology in [config.HEAT_PUMP, config.SOLAR_PV, config.EV_CHARGE_POINT]:
        first_reference = applications.loc[applications[config.TECHNOLOGY_COLUMN] == technology,
                                           config.APPLICATION_REFERENCE_COLUMN].iloc[0]
        templates[technology] = fetch_application(applications, first_reference)
    standards = {
        config.HEAT_PUMP: {"technology": config.HEAT_PUMP,
                           "document_name": "Invented-Assessment-Standard.pdf",
                           "requirements": [
                               {"requirement_kind": config.REGISTERED_EQUIPMENT_REQUIREMENT,
                                "requirement_name": "Registered equipment",
                                "requirement_sentence": ("The equipment is listed in the "
                                                         "approved register."), "rules": {}},
                               {"requirement_kind": config.SINGLE_CONTROLLER_REQUIREMENT,
                                "requirement_name": "Single controller",
                                "requirement_sentence": ("The equipment and every back-up or "
                                                         "boost heater are run by a single "
                                                         "controller."), "rules": {}}]},
        config.SOLAR_PV: {"technology": config.SOLAR_PV,
                          "document_name": "Invented-Assessment-Standard.pdf",
                          "requirements": [
                              {"requirement_kind": config.REGISTERED_EQUIPMENT_REQUIREMENT,
                               "requirement_name": "Registered equipment",
                               "requirement_sentence": ("The equipment is listed in the "
                                                        "type-test register."), "rules": {}},
                              {"requirement_kind": config.ISOLATION_REQUIREMENT,
                               "requirement_name": "Isolation",
                               "requirement_sentence": ("An isolator that locks off is fitted on "
                                                        "the inverter's mains side."),
                               "rules": {}}]},
    }
    listed, controller = config.REGISTER_LISTED_COLUMN, config.SINGLE_CONTROLLER_COLUMN
    looped, cut_out = config.LOOPED_SERVICE_COLUMN, config.CUT_OUT_RATING_COLUMN
    demand, rated = config.MAXIMUM_DEMAND_COLUMN, config.RATED_CURRENT_COLUMN
    inverter = config.INVERTER_CAPACITY_COLUMN
    answered_yes, answered_no = config.BOOLEAN_YES, config.BOOLEAN_NO
    approve, give_back = config.DECISION_APPROVE, config.DECISION_RETURN
    refer = config.DECISION_REFER
    looped_trigger, capacity = config.LOOPED_SERVICE_TRIGGER, config.SUPPLY_CAPACITY_TRIGGER
    generation = config.GENERATION_TRIGGER
    base = {listed: answered_yes, controller: answered_yes,
            config.ISOLATOR_FITTED_COLUMN: answered_yes, looped: answered_no, cut_out: 100,
            demand: 50.0, config.LOAD_LIMITER_FITTED_COLUMN: answered_no,
            config.LOAD_LIMITER_SETTING_COLUMN: None, config.SUPPLY_PHASES_COLUMN: 1,
            config.EQUIPMENT_PHASES_COLUMN: 1, rated: 16.0, inverter: 3.68,
            config.EXISTING_GENERATION_COLUMN: 0.0,
            config.EXPORT_LIMITATION_FITTED_COLUMN: answered_no}
    cases = [
        (config.HEAT_PUMP, {}, approve, [2, 0, 0], [], True),
        (config.HEAT_PUMP, {looped: answered_yes}, refer, [2, 0, 0], [looped_trigger], True),
        (config.HEAT_PUMP, {demand: 80.0, cut_out: 80}, refer, [2, 0, 0], [capacity], True),
        (config.HEAT_PUMP, {rated: None}, give_back, [2, 0, 0], [], False),
        (config.HEAT_PUMP, {listed: answered_no}, give_back, [1, 1, 0], [], True),
        (config.HEAT_PUMP, {listed: None, looped: answered_yes}, give_back, [1, 0, 1],
         [looped_trigger], True),
        (config.SOLAR_PV, {}, approve, [2, 0, 0], [], True),
        (config.SOLAR_PV, {inverter: 5.0}, refer, [2, 0, 0], [generation], True),
        (config.SOLAR_PV, {inverter: None}, give_back, [2, 0, 0], [generation], False),
    ]
    for technology, answers, expected_assessment, expected_counts, expected_triggers, stated in (
            cases):
        application = dict(templates[technology])
        application[config.APPLICATION_REFERENCE_COLUMN] = "LCT0950107"
        application.update(base)
        application.update(answers)
        screening = screen_application(application, standards[technology])
        assert isinstance(screening, dict), (
            "screen_application returned "
            + type(screening).__name__
            + ", where it returns the screening of the application."
        )
        counts = [screening.get(column_name) for column_name in config.REQUIREMENT_COUNT_RESULTS]
        assert (screening.get("assessment") == expected_assessment
                and counts == expected_counts
                and list(screening.get("network_triggers", [])) == expected_triggers
                and screening.get("rating_stated") is stated
                and list(screening.get("decisions_open", []))
                == config.DECISIONS_OPEN[expected_assessment]
                and screening.get("application_reference") == "LCT0950107"
                and screening.get("technology") == technology
                and screening.get("standard_document") == "Invented-Assessment-Standard.pdf"), (
            "A "
            + config.TECHNOLOGY_NAMES[technology]
            + " application with "
            + str(answers)
            + " on a form that otherwise meets every requirement and triggers nothing is "
            "assessed "
            + str(screening.get("assessment"))
            + " with counts "
            + str(counts)
            + ", the network triggers "
            + str(screening.get("network_triggers"))
            + ", the rating stated "
            + str(screening.get("rating_stated"))
            + " and the decisions open "
            + str(screening.get("decisions_open"))
            + ", where the Connection Rules give "
            + expected_assessment
            + ", "
            + str(expected_counts)
            + ", "
            + str(expected_triggers)
            + ", "
            + str(stated)
            + " and "
            + str(config.DECISIONS_OPEN[expected_assessment])
            + ". Any requirement Not met or Not documented, or a blank rating, makes the "
            "assessment RETURN TO INSTALLER before any network trigger counts, though the "
            "triggers are still found; otherwise a trigger makes it REFER FOR NETWORK "
            "ASSESSMENT; APPROVE CONNECTION alone leaves APPROVE CONNECTION open."
        )
    charge_point = templates[config.EV_CHARGE_POINT]
    refusal = None
    try:
        screen_application(charge_point, standards[config.HEAT_PUMP])
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    assert refusal is not None and charge_point[config.APPLICATION_REFERENCE_COLUMN] in refusal, (
        "An EV charge point application screened against a heat pump standard gave "
        + str(refusal)
        + ". An application is screened only against the standard of its own technology; any "
        "other is refused, naming the application."
    )
    written = live_files["raw_applications"].head(1).copy()
    for column_name, value in [(config.APPLICATION_REFERENCE_COLUMN, "41999"),
                               (config.REVISION_NUMBER_COLUMN, "1"),
                               (config.SOURCE_PORTAL_COLUMN, "SiteWren"),
                               (config.TECHNOLOGY_COLUMN, "HP"), (config.URGENCY_COLUMN, "STD"),
                               (config.SUPPLY_PHASES_COLUMN, "1"),
                               (config.EQUIPMENT_PHASES_COLUMN, "1"), (looped, "FALSE"),
                               (demand, "45"), (config.LOAD_LIMITER_FITTED_COLUMN, "FALSE"),
                               (config.LOAD_LIMITER_SETTING_COLUMN, "-"),
                               (config.EXPORT_LIMITATION_FITTED_COLUMN, "TRUE"),
                               (config.EXPORT_SCHEME_CERTIFIED_COLUMN, "TRUE"),
                               (config.EARTHING_ARRANGEMENT_COLUMN, "TT")]:
        written[column_name] = value
    worked_standard = {"technology": config.HEAT_PUMP,
                       "document_name": "Worked-Example-Standard.pdf",
                       "requirements": [
                           {"requirement_kind": config.RATED_CURRENT_REQUIREMENT,
                            "requirement_name": "Rated current",
                            "requirement_sentence": ("The rated current is at most 20 amps per "
                                                     "phase."),
                            "rules": {config.MAXIMUM_AMPS_RULE: Decimal("20")}},
                           {"requirement_kind": config.POWER_FACTOR_REQUIREMENT,
                            "requirement_name": "Power factor",
                            "requirement_sentence": ("The power factor is at least 0.90 at full "
                                                     "load."),
                            "rules": {config.MINIMUM_POWER_FACTOR_RULE: Decimal("0.90")}},
                           {"requirement_kind": config.EARTHING_PROTECTION_REQUIREMENT,
                            "requirement_name": "Earthing protection",
                            "requirement_sentence": ("On a PME earth the equipment has open-PEN "
                                                     "protection, and on a TT earth its earth "
                                                     "electrode is at most 200 ohms."),
                            "rules": {config.MAXIMUM_OHMS_RULE: Decimal("200")}},
                           {"requirement_kind": config.EXPORT_LIMITATION_REQUIREMENT,
                            "requirement_name": "Export limitation",
                            "requirement_sentence": ("An export limitation scheme, where one is "
                                                     "fitted, is certified and limits export to "
                                                     "at most 12 amps per phase."),
                            "rules": {config.MAXIMUM_AMPS_RULE: Decimal("12")}}]}
    electrode, power_factor = config.EARTH_ELECTRODE_COLUMN, config.POWER_FACTOR_COLUMN
    export_limit = config.EXPORT_LIMIT_COLUMN
    outcomes = {}
    for label, sent in [("as sent", {}),
                        ("with the resistance written 0.15 kilohms, the power factor 96% and "
                         "the cut-out 60", {electrode: "0.15k\u03a9", power_factor: "96%",
                                            cut_out: "60"}),
                        ("with the rating written 4.63 kW and the export limit 2,783 W",
                         {rated: "4.63 kW", export_limit: "2,783 W"})]:
        variant = written.copy()
        for column_name, value in [(rated, "4600W"), (power_factor, "89.5%"), (electrode, "-"),
                                   (export_limit, "2760W"), (cut_out, "unknown")]:
            variant[column_name] = sent.get(column_name, value)
        sent_path = str(tmp_path / "worked_example.csv")
        variant.to_csv(sent_path, index=False)
        cleaned = clean_application_records(load_raw_dataset(sent_path),
                                            config.FORM_VALUE_MAPPINGS,
                                            config.LIVE_WHOLE_NUMBER_PATTERNS)
        assert isinstance(cleaned, pd.DataFrame) and "LCT0041999" in set(
                cleaned.get(config.APPLICATION_REFERENCE_COLUMN, [])), (
            "clean_application_records returned "
            + str(cleaned)[:200]
            + " for the worked example written 41999, where it returns the cleaned application "
            "as LCT0041999."
        )
        cleaned_application = fetch_application(cleaned, "LCT0041999")
        screening = screen_application(cleaned_application, worked_standard)
        assert (isinstance(screening, dict) and isinstance(screening.get("requirements"), list)
                and all(isinstance(requirement, dict)
                        for requirement in screening["requirements"])), (
            "screen_application returned "
            + str(screening)[:200]
            + " for the worked example, where it returns its screening."
        )
        outcomes[label] = (screening.get("assessment"),
                           [requirement.get("result") for requirement in screening["requirements"]],
                           [list(requirement.get("blank_answers", []))
                            for requirement in screening["requirements"]],
                           list(screening.get("network_triggers", [])),
                           cleaned_application[rated], cleaned_application[power_factor],
                           cleaned_application[export_limit])
    met, not_met = config.REQUIREMENT_MET, config.REQUIREMENT_NOT_MET
    undocumented = config.REQUIREMENT_NOT_DOCUMENTED
    expected_outcomes = {
        "as sent": (give_back, [met, not_met, undocumented, met], [[], [], [electrode], []],
                    [capacity], 20.0, 0.895, 2.76),
        "with the resistance written 0.15 kilohms, the power factor 96% and the cut-out 60": (
            approve, [met, met, met, met], [[], [], [], []], [], 20.0, 0.96, 2.76),
        "with the rating written 4.63 kW and the export limit 2,783 W": (
            give_back, [not_met, not_met, undocumented, not_met], [[], [], [electrode], []],
            [capacity], 20.1, 0.895, 2.783),
    }
    assert outcomes == expected_outcomes, (
        "The worked example of the Connection Rules, section 13, screens as "
        + str(outcomes)
        + " where the Rules give "
        + str(expected_outcomes)
        + ": 4600W of single-phase equipment is exactly 20.0 amps, at the limit, so Met; 89.5% "
        "is 0.895, under 0.90, so Not met; a TT earth with its resistance written - is Not "
        "documented, asking for the resistance alone; an export limit of 2760W is exactly 12.0 "
        "amps, so Met; and the cut-out written unknown counts as reached, so the supply "
        "capacity trigger applies too, but the requirements come first."
    )


def test_agents_called_once_in_order(saved_run):
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    listed = saved_run["calls_at_gate"]
    assert (listed["status"] == 200 and isinstance(listed["body"], dict)
            and {"calls", "call_count", "expected_call_count"} <= set(listed["body"])
            and all("agent_name" in call for call in listed["body"]["calls"])), (
        "Reading the run's model calls at the pause returned "
        + str(listed["status"])
        + " with "
        + str(listed["body"])[:300]
        + ", where the audit area lists every call under the agent that made it, with their "
        "count and the count expected."
    )
    at_gate = listed["body"]
    reviewing = [config.AGENT_NAME_PLANNER, config.AGENT_NAME_REQUIREMENTS,
                 config.AGENT_NAME_EVIDENCE, config.AGENT_NAME_RISK, config.AGENT_NAME_WRITER]
    called = [call["agent_name"] for call in at_gate["calls"]]
    assert at_gate["call_count"] == len(at_gate["calls"]) == config.LLM_CALLS_BEFORE_DECISION, (
        "Reaching the Connections Engineer recorded a call count of "
        + str(at_gate["call_count"])
        + " and listed "
        + str(len(at_gate["calls"]))
        + " language model calls where the workflow allows "
        + str(config.LLM_CALLS_BEFORE_DECISION)
        + ", one per reviewing agent. More means an agent ran twice; fewer, that one never ran."
    )
    assert called == reviewing, (
        "The agents ran in the order "
        + str(called)
        + ". Each one reads what the one before it concluded: the requirements are explained "
        "against the plan, the statement evidence is reviewed against it, the figures are "
        "weighed with the plan in hand, and the recommendation draws on all of them."
    )
    assert at_gate["expected_call_count"] == config.LLM_CALLS_BEFORE_DECISION, (
        "Before any decision the service expected "
        + str(at_gate["expected_call_count"])
        + " calls. Until a decision is taken only the reviewing agents' calls are expected."
    )
    assert saved_run["at_gate"].get("llm_call_count") == config.LLM_CALLS_BEFORE_DECISION, (
        "The review record at the pause counts "
        + str(saved_run["at_gate"].get("llm_call_count"))
        + " calls where five were made."
    )


def test_model_calls_recorded_with_tokens(saved_run):
    from config import config
    from persistence.database import session_factory
    from persistence.models import LlmCall

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    final_calls = saved_run["final_calls"]
    assert final_calls["status"] == 200 and isinstance(final_calls["body"], dict), (
        "Reading the run's model calls after the decision returned "
        + str(final_calls["status"])
        + " with "
        + str(final_calls["body"])[:300]
        + "."
    )
    calls = final_calls["body"]["calls"]
    called = [call.get("agent_name") for call in calls]
    added_agents = [config.AGENT_NAME_INSTALLER_NOTICE, config.AGENT_NAME_HOUSEHOLD_NOTICE]
    expected_count = config.LLM_CALLS_BEFORE_DECISION + len(added_agents)
    assert (called[config.LLM_CALLS_BEFORE_DECISION:] == added_agents
            and final_calls["body"]["call_count"] == len(calls) == expected_count
            and final_calls["body"]["expected_call_count"]
            == config.LLM_CALLS_BY_DECISION[config.DECISION_APPROVE]), (
        "After the approval the run lists the calls "
        + str(called)
        + " ("
        + str(final_calls["body"].get("call_count"))
        + " of "
        + str(final_calls["body"].get("expected_call_count"))
        + " expected) where an approval adds the connection approval and then the household "
        "notice to the five reviewing calls."
    )
    for call in calls:
        tokens = [call.get("prompt_tokens"), call.get("response_tokens"), call.get("total_tokens")]
        assert (all(isinstance(count, int) and count > 0 for count in tokens)
                and tokens[2] >= tokens[0] + tokens[1]
                and isinstance(call.get("latency_milliseconds"), int)
                and call["latency_milliseconds"] >= 0
                and isinstance(call.get("model_name"), str) and call["model_name"]), (
            "The call of "
            + str(call.get("agent_name"))
            + " is recorded as "
            + str({key: call.get(key) for key in ["model_name", "prompt_tokens", "response_tokens",
                                                  "total_tokens", "latency_milliseconds"]})
            + ". Every call is recorded with the model that answered, the tokens it was sent, "
            "the tokens it wrote and the total the model counted, and the milliseconds it "
            "took, which is what it cost."
        )
    assert (final_calls["body"]["total_tokens"] == sum(call["total_tokens"] for call in calls)
            and final_calls["body"]["total_latency_milliseconds"]
            == sum(call["latency_milliseconds"] for call in calls)), (
        "The run's totals of "
        + str(final_calls["body"]["total_tokens"])
        + " tokens and "
        + str(final_calls["body"]["total_latency_milliseconds"])
        + " milliseconds are not the sums of its calls'."
    )
    with session_factory() as session:
        stored = session.query(LlmCall).filter(
            LlmCall.run_id == saved_run["at_gate"]["run_id"]).all()
    assert len(stored) == expected_count and sorted(
            row.agent_name for row in stored) == sorted(called), (
        "The llm_calls table holds "
        + str(len(stored))
        + " calls for the run, by "
        + str(sorted(row.agent_name for row in stored))
        + ", where the run made "
        + str(expected_count)
        + ", by "
        + str(sorted(called))
        + ". Each call is recorded once, under the run that made it."
    )


def test_recommendation_among_decisions_open(saved_run):
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    at_gate = saved_run["at_gate"]
    detail = saved_run["detail"]
    assert at_gate.get("decisions_open") == detail["screening"]["decisions_open"], (
        "The review offers the decisions "
        + str(at_gate.get("decisions_open"))
        + " where the screening of "
        + saved_run["application_reference"]
        + " leaves "
        + str(detail["screening"]["decisions_open"])
        + " open."
    )
    assert at_gate.get("recommended_decision") in at_gate.get("decisions_open", []), (
        "The review recommends "
        + repr(at_gate.get("recommended_decision"))
        + ", which is not among the decisions open "
        + str(at_gate.get("decisions_open"))
        + ". A recommendation the screening does not allow could never be taken."
    )
    write_ups = {key: str(at_gate.get(key) or "") for key in
                 ["review_plan_text", "requirements_analysis_text", "evidence_review_text",
                  "prediction_analysis_text", "recommendation_text"]}
    short = [key for key, text in write_ups.items() if len(text.split()) < 8]
    assert len(short) == 0, (
        "The review stores "
        + str(short)
        + " with fewer than eight words. Every reviewing agent's write-up is kept with the "
        "review for the Connections Engineer to read."
    )
    assert len(set(write_ups.values())) == len(write_ups), (
        "Two of the review's write-ups are the same text. Each agent writes its own part of "
        "the review."
    )
    assert (at_gate.get("coordinator_request") == saved_run["coordinator_request"]
            and at_gate.get("coordinator_id") == saved_run["opener"]["username"]), (
        "The review records the request "
        + repr(at_gate.get("coordinator_request"))
        + " opened by "
        + repr(at_gate.get("coordinator_id"))
        + " where "
        + saved_run["opener"]["username"]
        + " asked "
        + repr(saved_run["coordinator_request"])
        + "."
    )
    assert at_gate.get("status") == config.REVIEW_STATUS_AWAITING, (
        "The review was stored as "
        + repr(at_gate.get("status"))
        + " at the pause, where it waits as "
        + config.REVIEW_STATUS_AWAITING
        + "."
    )


def test_decorated_reply_read_whole():
    from agentic.agents import parse_labelled_response

    labels = ["PRIORITY", "CHECKS", "RATIONALE"]
    decorated = ("Here is the plan you asked for.\n"
                 "**PRIORITY:** The rated current requirement.\n"
                 "## Checks: 1. Confirm the rating against the wiring notes.\n"
                 "2. Confirm the register entry of the model.\n"
                 "- RATIONALE: The form decides; the statements are evidence.\n"
                 "rationale: The answers are read as the Connection Rules set out.")
    parsed = parse_labelled_response(decorated, labels, "review_planner_agent")
    assert (isinstance(parsed, dict) and sorted(parsed) == sorted(labels)
            and all(isinstance(value, str) for value in parsed.values())), (
        "Parsing a decorated reply returned "
        + str(parsed)[:300]
        + " where it returns exactly the fields "
        + str(labels)
        + ", each with its text."
    )
    assert (parsed["PRIORITY"] == "The rated current requirement."
            and parsed["CHECKS"] == ("1. Confirm the rating against the wiring notes.\n"
                                     "2. Confirm the register entry of the model.")
            and "The answers are read as the Connection Rules set out." in parsed["RATIONALE"]
            and "The form decides" in parsed["RATIONALE"]), (
        "A reply with labels in bold or as headings, a field running over two lines and a "
        "label written twice was read as "
        + str(parsed)
        + ". Decoration around a label is not part of it, a line with no label belongs to the "
        "field above it, and a label written again adds to its field."
    )
    for reply, missing in [("PRIORITY: The rating.\nCHECKS: The register.", "RATIONALE"),
                           ("PRIORITY: The rating.\nCHECKS:\nRATIONALE: Because.", "CHECKS")]:
        refusal = None
        try:
            parse_labelled_response(reply, labels, "review_planner_agent")
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and missing in refusal and "review_planner_agent" in refusal, (
            "A reply that leaves "
            + missing
            + " out or empty gave "
            + str(refusal)
            + ". A reply missing a field is refused with a ValueError naming the agent and the "
            "field, never stored with a gap."
        )


def test_approval_writes_both_notices(saved_run, live_files):
    from config import config
    from tools.application_data_tools import fetch_application
    from tools.notice_tools import build_approval_details

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    assert live_files["problem"] is None, live_files["problem"]
    final = saved_run["final_record"]
    assert final["status"] == 200 and len(final["body"]) == 1, (
        "The approved review could not be read back: "
        + str(final["body"])[:300]
    )
    record = final["body"][0]
    application = fetch_application(live_files["applications"], saved_run["application_reference"])
    built = build_approval_details(application, saved_run["detail"]["screening"])
    details = record.get("notice_details") or {}
    own_standard = config.CONNECTION_STANDARD_DOCUMENTS[application[config.TECHNOLOGY_COLUMN]]
    assert (isinstance(built, dict) and details == built
            and details.get(config.APPLICATION_REFERENCE_COLUMN)
            == saved_run["application_reference"]
            and details.get("standard_document") == own_standard), (
        "The connection approval carries the details "
        + str(details)
        + " where build_approval_details gives "
        + str(built)
        + " for the application's form and screening. The details are added from the form and "
        "the screening, never written by the model, and name the application and its own "
        "standard, "
        + own_standard
        + "."
    )
    notice = str(record.get("installer_notice_text") or "")
    household_notice = str(record.get("household_notice_text") or "")
    assert (len(notice.split()) >= 8 and len(household_notice.split()) >= 8
            and notice != household_notice), (
        "The approval stored a connection approval of "
        + str(len(notice.split()))
        + " words and a household notice of "
        + str(len(household_notice.split()))
        + ". An approval writes the connection approval to the installer and then a household "
        "notice to the people who live at the premises, two different texts."
    )
    assert record.get("assessment_brief") is None and record.get("status") == (
            config.REVIEW_STATUS_APPROVED), (
        "The approved review is stored as "
        + repr(record.get("status"))
        + " with an assessment brief of "
        + str(record.get("assessment_brief"))[:120]
        + ". An approval is recorded as "
        + config.REVIEW_STATUS_APPROVED
        + " and sends nothing to Network Planning."
    )


def test_workflow_nodes_and_decision_paths(tmp_path):
    from agentic.state import ApplicationReviewState
    from config import config

    try:
        from agentic.graph import build_workflow

        compiled = build_workflow(str(tmp_path / "workflow_check.db"))
    except Exception as error:
        compiled = error
    builder = getattr(compiled, "builder", None)
    assert all(hasattr(builder, name) for name in ["nodes", "branches", "edges"]) and callable(
            getattr(compiled, "get_state", None)), (
        "Building the workflow gave "
        + repr(compiled)[:200]
        + ", where it returns the compiled workflow, built from its nodes, edges and "
        "conditional branches and kept in a checkpoint store."
    )
    node_names = sorted(str(name) for name in builder.nodes)
    expected_nodes = sorted(["load_application_context", "plan_application_review",
                             "assess_connection_requirements", "analyse_installer_statements",
                             "interpret_model_predictions", "prepare_review_recommendation",
                             "await_engineer_decision", "draft_installer_notice",
                             "draft_household_notice", "record_connection_outcome"])
    assert node_names == expected_nodes, (
        "The workflow declares the nodes "
        + str(node_names)
        + " where the design has "
        + str(expected_nodes)
        + "."
    )
    destinations = {}
    for gate in ["await_engineer_decision", "draft_installer_notice"]:
        branches = builder.branches.get(gate, {})
        assert len(branches) == 1, (
            "The node "
            + gate
            + " has "
            + str(len(branches))
            + " conditional branches where it has one, choosing the path from the decision."
        )
        branch = list(branches.values())[0]
        decisions = config.DECISIONS
        if gate == "draft_installer_notice":
            decisions = [config.DECISION_APPROVE, config.DECISION_RETURN]
        for decision in decisions:
            state = {key: None for key in ApplicationReviewState.__annotations__}
            state["review_decision"] = {"decision": decision, "decided_by": "routing.check",
                                        "decided_by_role": config.ROLE_CONNECTIONS_ENGINEER,
                                        "decision_reason": "Routing is being checked."}
            routed = branch.path.invoke(state)
            if isinstance(routed, (list, tuple)) and len(routed) == 1:
                routed = routed[0]
            destinations[(gate, decision)] = branch.ends[routed] if branch.ends else routed
    expected_paths = {
        ("await_engineer_decision", config.DECISION_APPROVE): "draft_installer_notice",
        ("await_engineer_decision", config.DECISION_RETURN): "draft_installer_notice",
        ("await_engineer_decision", config.DECISION_REFER): "record_connection_outcome",
        ("draft_installer_notice", config.DECISION_APPROVE): "draft_household_notice",
        ("draft_installer_notice", config.DECISION_RETURN): "record_connection_outcome",
    }
    assert destinations == expected_paths, (
        "The decisions route to "
        + str(destinations)
        + ". An approval writes the connection approval and then the household notice; a "
        "return writes the rework request alone; a referral writes nothing and goes straight to "
        "the record, which adds the assessment brief."
    )
    edges = {(str(source), str(target)) for source, target in builder.edges}
    edges |= {(route[0], str(target)) for route, target in destinations.items()}
    reached = {"__start__"}
    grew = True
    while grew:
        grew = False
        for source, target in edges:
            if source in reached and target not in reached:
                reached.add(target)
                grew = True
    unreachable = sorted(set(node_names) - reached)
    assert len(unreachable) == 0, (
        "These nodes cannot be reached from the start: "
        + str(unreachable)
        + ". A node with no way in never runs, whatever it contains."
    )
    for source, target, reason in [
            ("__start__", "load_application_context",
             "every review starts from the tools' findings, before any agent runs"),
            ("prepare_review_recommendation", "await_engineer_decision",
             "the review is stored before the run waits for a Connections Engineer"),
            ("draft_household_notice", "record_connection_outcome",
             "what was written has to reach the record"),
            ("record_connection_outcome", "__end__",
             "the run ends once the outcome is recorded")]:
        assert (source, target) in edges, (
            "Nothing leads from " + source + " to " + target + ": " + reason + "."
        )


def test_run_waits_for_another_engineer(saved_run):
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    at_gate = saved_run["at_gate"]
    undecided = ["decision", "decided_by", "decided_by_role", "decision_reason",
                 "installer_notice_text", "notice_details", "household_notice_text",
                 "assessment_brief"]
    written = [field for field in undecided if at_gate.get(field) is not None]
    assert len(written) == 0 and at_gate.get("status") == config.REVIEW_STATUS_AWAITING, (
        "At the pause the review already holds "
        + str(written)
        + " with the status "
        + repr(at_gate.get("status"))
        + ". Nothing is decided, written or briefed until a Connections Engineer decides."
    )
    own = saved_run["own_decision"]
    expected_refusal = config.OWN_REVIEW_MESSAGE.format(
        decided_by=saved_run["opener"]["username"],
        application_reference=saved_run["application_reference"])
    assert (own["status"] == 409 and isinstance(own["body"], dict)
            and own["body"].get("detail") == expected_refusal), (
        "The Connections Engineer who opened the review, deciding it herself, was answered "
        + str(own["status"])
        + " with "
        + str(own["body"])[:300]
        + ", where the service refuses with 409 and the configuration's words: "
        + repr(expected_refusal)
        + ". The Low-Carbon Connections Policy has another Connections Engineer decide."
    )
    assert own["calls_before"] == own["calls_after"] == config.LLM_CALLS_BEFORE_DECISION, (
        "The refused decision moved the call count from "
        + str(own["calls_before"])
        + " to "
        + str(own["calls_after"])
        + ". A decision is checked before the run resumes, so a refused one spends no call."
    )


def test_run_resumes_from_checkpoint(saved_run):
    import os

    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    stored_run = saved_run["run_record"]
    assert stored_run["problem"] is None, stored_run["problem"]
    expected_calls = config.LLM_CALLS_BY_DECISION[config.DECISION_APPROVE]
    assert (stored_run["status"] == config.AGENT_RUN_STATUS_COMPLETE
            and stored_run["llm_call_count"] == expected_calls), (
        "After the approval the run is recorded as "
        + str(stored_run["status"])
        + " with "
        + str(stored_run["llm_call_count"])
        + " calls, where it closes "
        + config.AGENT_RUN_STATUS_COMPLETE
        + " with "
        + str(expected_calls)
        + "."
    )
    assert os.path.isfile(config.CHECKPOINT_DATABASE_FILE), (
        "There is no checkpoint at "
        + config.CHECKPOINT_DATABASE_FILE
        + ". A paused run waits on disk, so a decision taken later, or after the service "
        "restarts, finishes it."
    )
    try:
        from agentic.graph import build_workflow

        snapshot = build_workflow(config.CHECKPOINT_DATABASE_FILE).get_state(
            {"configurable": {"thread_id": saved_run["at_gate"]["thread_id"]}})
    except Exception as error:
        snapshot = error
    assert isinstance(getattr(snapshot, "values", None), dict) and hasattr(snapshot, "next"), (
        "Reading the run back through a workflow built afresh on "
        + config.CHECKPOINT_DATABASE_FILE
        + " gave "
        + repr(snapshot)[:200]
        + ", where it gives the run's state and what it waits for."
    )
    values = snapshot.values
    decision = values.get("review_decision") or {}
    assert (len(snapshot.next) == 0 and decision.get("decision") == config.DECISION_APPROVE
            and decision.get("decided_by") == saved_run["decider"]["username"]
            and values.get("llm_call_count") == expected_calls), (
        "A workflow built afresh on the checkpoint file reads the run as waiting for "
        + str(list(snapshot.next))
        + " with the decision "
        + str(decision)
        + " and "
        + str(values.get("llm_call_count"))
        + " calls. The run was resumed from that file and finished there, with the Connections "
        "Engineer's decision kept in its state."
    )
    assert (values.get("review_id") == saved_run["at_gate"]["review_id"]
            and values.get("application_reference") == saved_run["application_reference"]
            and stored_run["thread_id"] == saved_run["at_gate"]["thread_id"]), (
        "The checkpoint's state belongs to "
        + str(values.get("review_id"))
        + " and "
        + str(values.get("application_reference"))
        + ", not to the review that was decided."
    )


def test_decision_once_refusals_before_calls(saved_run, service_client):
    from config import config
    from persistence.database import session_factory
    from persistence.models import ReviewDecision
    from tools.decision_tools import find_decision_refusal, find_review_conflict

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    for name, meaning in [("short_reason", "a reason of five characters"),
                          ("unknown_decision", "a decision that is none of the three"),
                          ("short_request", "a request of two characters")]:
        assert saved_run[name]["status"] == 422, (
            "Sending "
            + meaning
            + " was answered "
            + str(saved_run[name]["status"])
            + " with "
            + str(saved_run[name]["body"])[:200]
            + ". The request's schema refuses it with 422 before anything is looked up."
        )
    second = saved_run["second_review"]
    expected_second = config.APPLICATION_ALREADY_REVIEWED_MESSAGE.format(
        application_reference=saved_run["application_reference"])
    assert (second["status"] == 409 and second["body"].get("detail") == expected_second
            and saved_run["run_record"].get("runs_opened") == 1), (
        "Opening a second review of "
        + saved_run["application_reference"]
        + " was answered "
        + str(second["status"])
        + " with "
        + str(second["body"])[:200]
        + ", and this run of the suite recorded "
        + str(saved_run["run_record"].get("runs_opened"))
        + " runs of the application. An application in the queue is reviewed once, and a "
        "second review is refused before a run starts: "
        + repr(expected_second)
    )
    decided = saved_run["decided"]["body"]
    assert (decided.get("decision") == config.DECISION_APPROVE
            and decided.get("decided_by") == saved_run["decider"]["username"]
            and decided.get("decided_by_role") == config.ROLE_CONNECTIONS_ENGINEER
            and decided.get("decision_reason") == saved_run["decision_reason"]
            and decided.get("status") == config.REVIEW_STATUS_APPROVED), (
        "The approval came back recorded as "
        + str({key: decided.get(key) for key in ["decision", "decided_by", "decided_by_role",
                                                 "decision_reason", "status"]})
        + ", which is not the decision the Connections Engineer sent."
    )
    review_id = saved_run["at_gate"]["review_id"]
    again = service_client["client"].post(
        "/reviews/" + review_id + "/decision",
        json={"decision": config.DECISION_APPROVE, "decision_reason": saved_run["decision_reason"]},
        headers=saved_run["decider"]["headers"])
    expected_again = config.ALREADY_DECIDED_MESSAGE.format(
        application_reference=saved_run["application_reference"])
    assert again.status_code == 409 and again.json().get("detail") == expected_again, (
        "Deciding the review a second time was answered "
        + str(again.status_code)
        + " with "
        + again.text[:200]
        + ". A review is decided once: "
        + repr(expected_again)
    )
    with session_factory() as session:
        stored = session.query(ReviewDecision).filter(ReviewDecision.review_id == review_id).all()
    assert len(stored) == 1 and stored[0].decision == config.DECISION_APPROVE, (
        "The review_decisions table holds "
        + str(len(stored))
        + " decisions for the review where it holds the one decision taken."
    )
    reference = "LCT0950107"
    sound = {"decision": config.DECISION_APPROVE, "decided_by": "lewis",
             "decided_by_role": config.ROLE_CONNECTIONS_ENGINEER,
             "decision_reason": "Every requirement is met on the form."}
    approved_screening = {"application_reference": reference,
                          "assessment": config.DECISION_APPROVE,
                          "decisions_open": config.DECISIONS_OPEN[config.DECISION_APPROVE]}
    cases = [
        ("an approval of an application assessed RETURN TO INSTALLER", sound,
         dict(approved_screening, assessment=config.DECISION_RETURN,
              decisions_open=config.DECISIONS_OPEN[config.DECISION_RETURN]),
         config.DECISION_NOT_OPEN_MESSAGE.format(
             decision=config.DECISION_APPROVE, application_reference=reference,
             reason=config.APPROVE_CLOSED_REASONS[config.DECISION_RETURN])),
        ("an approval of an application assessed REFER FOR NETWORK ASSESSMENT", sound,
         dict(approved_screening, assessment=config.DECISION_REFER,
              decisions_open=config.DECISIONS_OPEN[config.DECISION_REFER]),
         config.DECISION_NOT_OPEN_MESSAGE.format(
             decision=config.DECISION_APPROVE, application_reference=reference,
             reason=config.APPROVE_CLOSED_REASONS[config.DECISION_REFER])),
        ("a decision with no reason", dict(sound, decision_reason="  "), approved_screening,
         config.DECISION_FIELDS_MISSING_MESSAGE.format(
             application_reference=reference, fields="decision_reason")),
        ("a decision that is none of the three", dict(sound, decision="CONNECT"),
         approved_screening, config.UNKNOWN_DECISION_MESSAGE.format(
             decision="CONNECT", application_reference=reference,
             decisions=", ".join(config.DECISIONS))),
        ("a decision by a Connections Coordinator",
         dict(sound, decided_by_role=config.ROLE_CONNECTIONS_COORDINATOR), approved_screening,
         config.DECIDING_ROLE_MESSAGE.format(
             role=config.ROLE_CONNECTIONS_COORDINATOR, application_reference=reference,
             engineer_role=config.ROLE_CONNECTIONS_ENGINEER)),
        ("a decision by the engineer who opened the review", dict(sound, decided_by="megan"),
         approved_screening, config.OWN_REVIEW_MESSAGE.format(
             decided_by="megan", application_reference=reference)),
        ("a referral of an application assessed APPROVE CONNECTION",
         dict(sound, decision=config.DECISION_REFER), approved_screening, None),
        ("a sound approval", sound, approved_screening, None),
    ]
    for meaning, review_decision, screening, expected_refusal in cases:
        refusal = find_decision_refusal(review_decision, screening, "megan")
        assert refusal == expected_refusal, (
            "find_decision_refusal answered "
            + meaning
            + " with "
            + repr(refusal)
            + " where it answers "
            + repr(expected_refusal)
            + ". Every decision the workflow's gate would refuse is named before the run "
            "resumes, in the configuration's words, so no call is spent on it."
        )
    other = dict(sound, decision=config.DECISION_RETURN, decided_by="amira")
    for meaning, review_status, pending, expected_conflict in [
            ("a decision on a review already approved", config.REVIEW_STATUS_APPROVED, None,
             config.ALREADY_DECIDED_MESSAGE.format(application_reference=reference)),
            ("an approval while amira's return is still being recorded",
             config.REVIEW_STATUS_AWAITING, other,
             config.DECISION_IN_PROGRESS_MESSAGE.format(
                 pending_decided_by="amira", pending_decision=config.DECISION_RETURN,
                 application_reference=reference)),
            ("the same decision sent again to finish it", config.REVIEW_STATUS_AWAITING, sound,
             None),
            ("a decision on a review awaiting one", config.REVIEW_STATUS_AWAITING, None, None)]:
        conflict = find_review_conflict(review_status, pending, sound, reference)
        assert conflict == expected_conflict, (
            "find_review_conflict answered "
            + meaning
            + " with "
            + repr(conflict)
            + " where it answers "
            + repr(expected_conflict)
            + ". A review is decided once, and a decision still being recorded is finished only "
            "by the same decision sent again."
        )


def test_stored_review_matches_tools(saved_run, saved_models, connection_standards, live_files):
    from config import config
    from tools.application_data_tools import fetch_application
    from tools.prediction_tools import (estimate_expected_working_days,
                                        estimate_network_work_likelihood)
    from tools.screening_tools import screen_application

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert connection_standards["problem"] is None, connection_standards["problem"]
    assert live_files["problem"] is None, live_files["problem"]
    record = saved_run["final_record"]["body"][0]
    detail = saved_run["detail"]
    screening = detail["screening"]
    pairs = {
        "requirements": (record.get("requirements"), screening["requirements"]),
        "assessment": (record.get("assessment"), screening["assessment"]),
        "decisions_open": (record.get("decisions_open"), screening["decisions_open"]),
        "network_triggers": (record.get("network_triggers"), screening["network_triggers"]),
        "rating_stated": (record.get("rating_stated"), screening["rating_stated"]),
        "technology": (record.get("technology"), screening["technology"]),
        "standard_document": (record.get("standard_document"), screening["standard_document"]),
        "evidence": (record.get("evidence"), detail["evidence_findings"]),
        "disagreements": (record.get("disagreements"), detail["disagreements"]),
        "network_work_likelihood": (record.get("network_work_likelihood"),
                                    detail["likelihood"]["network_work_likelihood"]),
        "likelihood_band": (record.get("likelihood_band"),
                            detail["likelihood"]["likelihood_band"]),
        "expected_working_days": (record.get("expected_working_days"),
                                  detail["duration"]["expected_working_days"]),
        "planned_working_days": (record.get("planned_working_days"),
                                 detail["duration"]["planned_working_days"]),
        "delay_expected": (record.get("delay_expected"), detail["duration"]["delay_expected"]),
        "duration_text": (record.get("duration_text"), detail["duration_text"]),
    }
    for count_column in config.REQUIREMENT_COUNT_RESULTS:
        pairs[count_column] = (record.get(count_column), screening[count_column])
    differing = {name: pair for name, pair in pairs.items() if pair[0] != pair[1]}
    assert len(differing) == 0, (
        "The stored review differs from the tools' findings for "
        + saved_run["application_reference"]
        + " in "
        + str({name: (str(pair[0])[:80], str(pair[1])[:80]) for name, pair in
               differing.items()})
        + ". The review records the screening, the statement evidence, the flags, the "
        "likelihood and the expected working days the tools worked out, never the agents' "
        "retelling of them."
    )
    application = fetch_application(live_files["applications"], saved_run["application_reference"])
    own_screening = screen_application(application, connection_standards["standards"][
        application[config.TECHNOLOGY_COLUMN]])
    likelihood = estimate_network_work_likelihood(
        saved_models["network_work"], saved_models["network_work_schema"], application,
        own_screening)
    duration = estimate_expected_working_days(
        saved_models["duration"], saved_models["duration_schema"], application, own_screening)
    assert (isinstance(own_screening, dict) and isinstance(likelihood, dict)
            and isinstance(duration, dict)), (
        "Worked out again for "
        + saved_run["application_reference"]
        + ", the screening, the likelihood and the expected working days came back as "
        + type(own_screening).__name__
        + ", "
        + type(likelihood).__name__
        + " and "
        + type(duration).__name__
        + ", where each is a record of its findings."
    )
    assert (likelihood.get("network_work_likelihood") == record.get("network_work_likelihood")
            and duration.get("expected_working_days") == record.get("expected_working_days")
            and own_screening.get("assessment") == record.get("assessment")
            and own_screening.get("network_triggers") == record.get("network_triggers")), (
        "Worked out again from the live files, the saved models and the application's own "
        "standard, "
        + saved_run["application_reference"]
        + " is assessed "
        + str(own_screening.get("assessment"))
        + " with the triggers "
        + str(own_screening.get("network_triggers"))
        + ", a likelihood of "
        + str(likelihood.get("network_work_likelihood"))
        + " and "
        + str(duration.get("expected_working_days"))
        + " expected working days, where the review stored "
        + str(record.get("assessment"))
        + ", "
        + str(record.get("network_triggers"))
        + ", "
        + str(record.get("network_work_likelihood"))
        + " and "
        + str(record.get("expected_working_days"))
        + "."
    )


def test_sign_in_and_out_decide_access(service_client):
    from access.password_security import hash_password
    from config import config
    from persistence.database import session_factory
    from persistence.models import UserAccount
    from persistence.repository import fetch_user_account

    assert service_client["problem"] is None, service_client["problem"]
    client = service_client["client"]
    coordinators = [account for account in config.SEED_USER_ACCOUNTS
                    if account["role"] == config.ROLE_CONNECTIONS_COORDINATOR]
    account = coordinators[1]
    other_account = coordinators[2]
    credentials = {"username": account["username"], "password": account["password"]}
    signed_in = client.post("/auth/signin", json=credentials)
    body = signed_in.json() if signed_in.status_code == 200 else {}
    assert ({"username", "display_name", "role", "session_token"} <= set(body)
            and len(str(body["session_token"]).strip()) > 0), (
        "Signing in as "
        + account["username"]
        + " with the password the configuration seeds returned "
        + str(signed_in.status_code)
        + " with "
        + signed_in.text[:200]
        + ", where a sign-in answers with the user's name, display name, role and a session "
        "token."
    )
    assert (body["username"] == account["username"] and body["role"] == account["role"]
            and body["display_name"] == account["display_name"]), (
        "Signing in as "
        + account["username"]
        + " answered with the name "
        + repr(body["username"])
        + ", the display name "
        + repr(body["display_name"])
        + " and the role "
        + repr(body["role"])
        + ", which are not those of the account the configuration seeds."
    )
    wrong = client.post("/auth/signin", json={"username": account["username"],
                                              "password": "wrong-" + account["password"]})
    stranger = client.post("/auth/signin", json={"username": "no-such-person",
                                                 "password": account["password"]})
    assert (wrong.status_code == 401 and stranger.status_code == 401
            and wrong.json() == stranger.json() == {"detail": config.SIGN_IN_FAILED_MESSAGE}), (
        "A wrong password was answered "
        + str(wrong.status_code)
        + " with "
        + wrong.text[:200]
        + " and an unknown username "
        + str(stranger.status_code)
        + " with "
        + stranger.text[:200]
        + ". Both are refused with 401 and the same words, "
        + repr(config.SIGN_IN_FAILED_MESSAGE)
        + ", so a stranger cannot tell which usernames exist."
    )
    parameters = {"technology": config.QUEUE_TECHNOLOGY_ALL}
    first_headers = {config.SESSION_TOKEN_HEADER: body["session_token"]}
    accepted = client.get("/queue", params=parameters, headers=first_headers)
    assert accepted.status_code == 200, (
        "The token just given to "
        + account["username"]
        + " was answered "
        + str(accepted.status_code)
        + " by the queue. A fresh session token opens the areas its role is granted."
    )
    for meaning, headers in [("with no token", None),
                             ("with a token nobody was given",
                              {config.SESSION_TOKEN_HEADER: "not-a-token-anyone-was-given"})]:
        refused = client.get("/queue", params=parameters, headers=headers)
        assert (refused.status_code == 401
                and refused.json() == {"detail": config.SIGN_IN_REQUIRED_MESSAGE}), (
            "The queue answered a request "
            + meaning
            + " with "
            + str(refused.status_code)
            + " and "
            + refused.text[:200]
            + ". Only a signed-in user is served; anyone else is refused with 401 and "
            + repr(config.SIGN_IN_REQUIRED_MESSAGE)
            + "."
        )
    again = client.post("/auth/signin", json=credentials)
    second_token = again.json().get("session_token") if again.status_code == 200 else None
    second_headers = {config.SESSION_TOKEN_HEADER: str(second_token)}
    replaced = client.get("/queue", params=parameters, headers=first_headers)
    current = client.get("/queue", params=parameters, headers=second_headers)
    assert (again.status_code == 200 and replaced.status_code == 401
            and current.status_code == 200), (
        "After "
        + account["username"]
        + " signed in a second time, answered "
        + str(again.status_code)
        + ", the first token got "
        + str(replaced.status_code)
        + " from the queue and the second "
        + str(current.status_code)
        + ". An account holds one session: signing in somewhere else ends the session before "
        "it."
    )
    signed_out = client.post("/auth/signout", headers=second_headers)
    after = client.get("/queue", params=parameters, headers=second_headers)
    twice = client.post("/auth/signout", headers=second_headers)
    assert (signed_out.status_code == 200
            and signed_out.json() == {"username": account["username"], "signed_out": True}
            and after.status_code == 401 and twice.status_code == 401), (
        "Signing out answered "
        + str(signed_out.status_code)
        + " with "
        + signed_out.text[:200]
        + "; the token then got "
        + str(after.status_code)
        + " from the queue and "
        + str(twice.status_code)
        + " when it signed out again. Signing out answers with the username and signed_out "
        "true, and the token it ends opens nothing afterwards."
    )
    with session_factory() as session:
        stored = fetch_user_account(session, UserAccount.username == account["username"])
        other = fetch_user_account(session, UserAccount.username == other_account["username"])
    missing = sorted({"password_hash", "password_salt"}
                     - (set(stored if isinstance(stored, dict) else {})
                        & set(other if isinstance(other, dict) else {})))
    assert len(missing) == 0, (
        "fetch_user_account read "
        + account["username"]
        + " and "
        + other_account["username"]
        + " back without "
        + str(missing)
        + ", so how their passwords are kept cannot be read."
    )
    assert (stored["password_hash"] != account["password"]
            and stored["password_hash"] == hash_password(account["password"],
                                                         stored["password_salt"])), (
        "The password stored for "
        + account["username"]
        + " is not the salted hash of the password. A password kept as written is read by "
        "anyone who can open the database."
    )
    assert (stored["password_salt"] != other["password_salt"]
            and stored["password_hash"] != other["password_hash"]), (
        account["username"]
        + " and "
        + other_account["username"]
        + " share a password and are stored with the same salt or the same hash, so one "
        "cracked hash opens both accounts."
    )


def test_roles_reach_only_granted_areas(service_client):
    from config import config

    assert service_client["problem"] is None, service_client["problem"]
    client = service_client["client"]
    signed_in = {}
    for role_name, position in [(config.ROLE_CONNECTIONS_ENGINEER, 1),
                                (config.ROLE_CONNECTIONS_COORDINATOR, 3),
                                (config.ROLE_CUSTOMER_SERVICE_ADVISOR, 1)]:
        accounts = [account for account in config.SEED_USER_ACCOUNTS
                    if account["role"] == role_name]
        account = accounts[position]
        reply = client.post("/auth/signin", json={"username": account["username"],
                                                  "password": account["password"]})
        body = reply.json() if reply.status_code == 200 else {}
        assert {"session_token", "pages", "areas"} <= set(body), (
            "Signing in as "
            + account["username"]
            + " returned "
            + str(reply.status_code)
            + " with "
            + reply.text[:200]
            + ", where a sign-in answers with a session token and the pages and areas of the "
            "user's role."
        )
        assert (sorted(body["pages"]) == sorted(config.ROLE_PAGE_ACCESS[role_name])
                and sorted(body["areas"]) == sorted(config.ROLE_AREA_ACCESS[role_name])), (
            "Signing in as a "
            + role_name
            + " answered with the pages "
            + str(body["pages"])
            + " and the areas "
            + str(body["areas"])
            + ", where the configuration grants that role the pages "
            + str(config.ROLE_PAGE_ACCESS[role_name])
            + " and the areas "
            + str(config.ROLE_AREA_ACCESS[role_name])
            + "."
        )
        signed_in[role_name] = {config.SESSION_TOKEN_HEADER: body["session_token"]}
    unknown_review = config.REVIEW_ID_PREFIX + "DOES-NOT-EXIST"
    probes = [
        (config.AREA_QUEUE, "GET", "/queue", {"technology": config.QUEUE_TECHNOLOGY_ALL},
         None, 200),
        (config.AREA_QUEUE, "GET", "/queue/LCT0000000", None, None, 404),
        (config.AREA_REVIEW, "POST", "/reviews", None,
         {"application_reference": "LCT0000000",
          "coordinator_request": "Are the requirements met on the form?"}, 404),
        (config.AREA_DECISION, "POST", "/reviews/" + unknown_review + "/decision", None,
         {"decision": config.DECISION_REFER,
          "decision_reason": "A network trigger applies to this application."}, 404),
        (config.AREA_AUDIT, "GET", "/reviews",
         {"status_value": [config.REVIEW_STATUS_AWAITING]}, None, 200),
        (config.AREA_AUDIT, "GET", "/reviews/" + unknown_review + "/calls", None, None, 404),
        (config.AREA_AUDIT, "GET", "/audit/chat-turns", {"turn_limit": 1}, None, 200),
        (config.AREA_CHAT, "GET", "/chat/turns", {"turn_limit": 1}, None, 200),
    ]
    for role_name, role_headers in signed_in.items():
        for area_name, method, path, parameters, payload, status_when_granted in probes:
            reply = client.request(method, path, params=parameters, json=payload,
                                   headers=role_headers)
            granted = area_name in config.ROLE_AREA_ACCESS[role_name]
            if granted:
                assert reply.status_code == status_when_granted, (
                    "A signed-in "
                    + role_name
                    + " calling "
                    + method
                    + " "
                    + path
                    + " got "
                    + str(reply.status_code)
                    + " with "
                    + reply.text[:200]
                    + " where it gets "
                    + str(status_when_granted)
                    + ". The "
                    + area_name
                    + " area is granted to that role, so the guard lets it through and the "
                    "endpoint answers: an application or a review that does not exist is "
                    "answered 404, before any language model call."
                )
            else:
                expected_refusal = config.AREA_REFUSED_MESSAGE.format(role=role_name,
                                                                      area=area_name)
                assert (reply.status_code == 403
                        and reply.json().get("detail") == expected_refusal), (
                    "A signed-in "
                    + role_name
                    + " calling "
                    + method
                    + " "
                    + path
                    + " got "
                    + str(reply.status_code)
                    + " with "
                    + reply.text[:200]
                    + ". The "
                    + area_name
                    + " area is not granted to that role, so the guard refuses it with 403 and "
                    "the configuration's words, "
                    + repr(expected_refusal)
                    + ", whether or not the thing asked for exists."
                )


def test_queue_order_and_rows_match_detail(
        service_client, staff, live_files, saved_models, connection_standards):
    from config import config
    from tools.application_data_tools import fetch_application, rank_application_queue
    from tools.prediction_tools import (estimate_expected_working_days,
                                        estimate_network_work_likelihood)
    from tools.screening_tools import screen_application

    assert staff["problem"] is None, staff["problem"]
    assert live_files["problem"] is None, live_files["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert connection_standards["problem"] is None, connection_standards["problem"]
    heating_failed, standard = config.URGENCY_HEATING_FAILED, config.URGENCY_STANDARD
    planted = live_files["applications"].head(5).copy()
    planted[config.APPLICATION_REFERENCE_COLUMN] = ["LCT0500004", "LCT0500002", "LCT0500001",
                                                    "LCT0500003", "LCT0500000"]
    planted[config.TECHNOLOGY_COLUMN] = [config.HEAT_PUMP, config.HEAT_PUMP,
                                         config.EV_CHARGE_POINT, config.HEAT_PUMP,
                                         config.HEAT_PUMP]
    planted[config.URGENCY_COLUMN] = [standard, heating_failed, heating_failed, heating_failed,
                                      None]
    ranked = rank_application_queue(planted)
    ranked = list(ranked) if isinstance(ranked, (list, tuple)) else ranked
    assert ranked == ["LCT0500002", "LCT0500003", "LCT0500000", "LCT0500001", "LCT0500004"], (
        "rank_application_queue put planted applications - a standard heat pump LCT0500004, "
        "heat pumps replacing failed heating LCT0500002 and LCT0500003, an EV charge point "
        "LCT0500001 marked as replacing failed heating and a heat pump LCT0500000 with its "
        "urgency blank - in the order "
        + str(ranked)
        + ". Heat pumps replacing a failed heating system come first, the rest after them, and "
        "each group runs by application reference."
    )
    client = service_client["client"]
    headers = staff["coordinator"]["headers"]
    reply = client.get("/queue", params={"technology": config.QUEUE_TECHNOLOGY_ALL},
                       headers=headers)
    assert reply.status_code == 200 and isinstance(reply.json(), list), (
        "The queue returned "
        + str(reply.status_code)
        + " with "
        + reply.text[:200]
        + " to a signed-in Connections Coordinator."
    )
    everything = reply.json()
    missing = sorted({"queue_position", "application_reference", "technology",
                      "technology_label", "source_portal", "premises_type", "urgency",
                      "document_count", "requirements_met_count", "requirements_not_met_count",
                      "requirements_not_documented_count", "network_trigger_count",
                      "rating_stated", "assessment", "decisions_open", "network_work_likelihood",
                      "likelihood_band", "expected_working_days", "delay_expected",
                      "disagreement_count"}
                     - set(everything[0] if everything else {}))
    assert len(missing) == 0, (
        "A queue row carries no "
        + str(missing)
        + ", so the queue cannot be ordered, filtered or read."
    )
    references = [row["application_reference"] for row in everything]
    applications = live_files["applications"]
    open_references = list(applications[config.APPLICATION_REFERENCE_COLUMN])
    assert len(set(references)) == len(references) and sorted(references) == sorted(
            open_references), (
        "The queue holds "
        + str(len(references))
        + " rows for "
        + str(len(set(references)))
        + " applications, where the open applications, read through the cleaning, number "
        + str(len(open_references))
        + ". Every open application is in the queue exactly once, at its latest revision."
    )
    failed_heating = set(applications.loc[
        (applications[config.TECHNOLOGY_COLUMN] == config.HEAT_PUMP)
        & (applications[config.URGENCY_COLUMN] == heating_failed),
        config.APPLICATION_REFERENCE_COLUMN])
    flags = [row["application_reference"] in failed_heating for row in everything]
    first_other = flags.index(False) if False in flags else len(flags)
    first_block, other_block = references[:first_other], references[first_other:]
    assert (not any(flags[first_other:]) and len(failed_heating) > 0
            and set(first_block) == failed_heating
            and first_block == sorted(first_block)
            and other_block == sorted(other_block)), (
        "The queue opens with "
        + str(first_block[:6])
        + " and then runs "
        + str(other_block[:6])
        + " ..., where the open heat pumps replacing a failed heating system are "
        + str(sorted(failed_heating))
        + ". They come first and then the rest, each group in the order of its application "
        "references, so the Connections Coordinators work the queue from the top."
    )
    assert [row["queue_position"] for row in everything] == list(range(1, len(everything) + 1)), (
        "The queue positions run "
        + str([row["queue_position"] for row in everything][:10])
        + " ... where they count from 1 down the queue."
    )
    checked_against_tools = [references[0], references[len(references) // 2], references[-1]]
    requirement_keys = ["requirement_kind", "requirement_name", "requirement_sentence", "result",
                        "blank_answers"]
    differing = {}
    for row in everything:
        detail_reply = client.get("/queue/" + row["application_reference"], headers=headers)
        detail = detail_reply.json() if detail_reply.status_code == 200 else {}
        described = detail.get("application", {})
        shown = detail.get("screening", {})
        shown_likelihood = detail.get("likelihood", {})
        shown_duration = detail.get("duration", {})
        expected = {
            "queue_position": detail.get("queue_position"),
            "technology": shown.get("technology"),
            "technology_label": config.TECHNOLOGY_LABELS.get(shown.get("technology")),
            "source_portal": described.get(config.SOURCE_PORTAL_COLUMN),
            "premises_type": described.get(config.PREMISES_TYPE_COLUMN),
            "urgency": described.get(config.URGENCY_COLUMN),
            "document_count": len(detail.get("application_documents", [])),
            "requirements_met_count": shown.get(config.REQUIREMENTS_MET_COUNT_COLUMN),
            "requirements_not_met_count": shown.get(config.REQUIREMENTS_NOT_MET_COUNT_COLUMN),
            "requirements_not_documented_count": shown.get(
                config.REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN),
            "network_trigger_count": len(shown.get("network_triggers", [])),
            "rating_stated": shown.get("rating_stated"),
            "assessment": shown.get("assessment"),
            "decisions_open": shown.get("decisions_open"),
            "network_work_likelihood": shown_likelihood.get("network_work_likelihood"),
            "likelihood_band": shown_likelihood.get("likelihood_band"),
            "expected_working_days": shown_duration.get("expected_working_days"),
            "delay_expected": shown_duration.get("delay_expected"),
            "disagreement_count": len(detail.get("disagreements", [])),
        }
        for key, value in expected.items():
            if row[key] != value:
                differing[row["application_reference"]] = (key + " is " + str(row[key])[:60]
                                                           + " where its detail gives "
                                                           + str(value)[:60])
        if row["application_reference"] in checked_against_tools:
            application = fetch_application(applications, row["application_reference"])
            screening = screen_application(application, connection_standards["standards"][
                application[config.TECHNOLOGY_COLUMN]])
            assert isinstance(screening, dict) and isinstance(
                    screening.get("requirements"), list), (
                "screen_application returned "
                + str(screening)[:200]
                + " for "
                + row["application_reference"]
                + ", where it returns the application's screening."
            )
            for key in ["technology", "standard_document", "network_triggers", "rating_stated",
                        "assessment", "decisions_open"] + list(config.REQUIREMENT_COUNT_RESULTS):
                if shown.get(key) != screening.get(key):
                    differing[row["application_reference"]] = ("the detail's " + key
                                                               + " is not the tools' own")
            own_requirements = [{key: requirement.get(key) for key in requirement_keys}
                                for requirement in screening["requirements"]
                                if isinstance(requirement, dict)]
            if shown.get("requirements") != own_requirements:
                differing[row["application_reference"]] = ("the detail's requirements are not "
                                                           "the tools'")
            likelihood = estimate_network_work_likelihood(
                saved_models["network_work"], saved_models["network_work_schema"], application,
                screening)
            duration = estimate_expected_working_days(
                saved_models["duration"], saved_models["duration_schema"], application, screening)
            if shown_likelihood != likelihood or shown_duration != duration:
                differing[row["application_reference"]] = ("the detail's figures are not the "
                                                           "saved models'")
    assert len(differing) == 0, (
        str(len(differing))
        + " rows of the queue differ from their detail or from the tools, for example "
        + str(dict(list(differing.items())[:3]))
        + ". Each row shows what its detail shows: its application's own screening under its "
        "technology's standard, the figures the saved models give it, the statements attached "
        "to it and the equipment ratings flagged in them."
    )
    for technology in config.QUEUE_TECHNOLOGY_OPTIONS[1:]:
        filtered_reply = client.get("/queue", params={"technology": technology}, headers=headers)
        filtered = filtered_reply.json() if filtered_reply.status_code == 200 else []
        assert len(filtered) > 0 and filtered == [row for row in everything
                                                  if row["technology"] == technology], (
            "Filtering the queue to "
            + technology
            + " did not return exactly the "
            + technology
            + " applications of the full queue, in the same order and with the same positions."
        )
    for meaning, parameters in [("a technology Merrowdale has no standard for",
                                 {"technology": "WT"}), ("no technology at all", None)]:
        refused = client.get("/queue", params=parameters, headers=headers)
        assert refused.status_code == 422, (
            "Asking for the queue with "
            + meaning
            + " returned "
            + str(refused.status_code)
            + " rather than 422. The filter takes a technology the configuration names, or all "
            "of them, and nothing else."
        )
    unknown = client.get("/queue/LCT0000000", headers=headers)
    expected_unknown = config.UNKNOWN_APPLICATION_MESSAGE.format(
        application_reference="LCT0000000")
    assert unknown.status_code == 404 and unknown.json().get("detail") == expected_unknown, (
        "The detail of LCT0000000, which is not an open application, returned "
        + str(unknown.status_code)
        + " with "
        + unknown.text[:200]
        + " where it is 404 with "
        + repr(expected_unknown)
        + "."
    )


def test_chat_answers_from_documents_with_memory(chat_replies, knowledge_collection, tmp_path):
    import os

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from chat.conversation_memory import recall_conversation
    from config import config
    from knowledge.vector_index import search_documents
    from persistence.database import session_factory
    from persistence.models import Base
    from persistence.repository import insert_chat_turn

    assert chat_replies["problem"] is None, chat_replies["problem"]
    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    for name, meaning in [("standard", "Asking the connection assistant what the heat pump "
                                       "connection standard asks for"),
                          ("turns", "Reading back the Connections Coordinator's own chat turns")]:
        assert chat_replies[name]["status"] == 200, (
            meaning
            + " returned "
            + str(chat_replies[name]["status"])
            + " with "
            + str(chat_replies[name]["body"])[:300]
            + "."
        )
    reply = chat_replies["standard"]["body"]
    missing = sorted({"question_type", "application_reference", "substation_reference",
                      "technology", "answer", "source_documents", "was_refused"} - set(reply))
    assert len(missing) == 0, (
        "The assistant's reply carries no "
        + str(missing)
        + ", so what it answered and where from cannot be read."
    )
    assert (reply["was_refused"] is False and reply["question_type"] in config.CHAT_ANSWERED_TYPES
            and reply["application_reference"] is None
            and reply["substation_reference"] is None), (
        "Asking what the heat pump connection standard asks for was answered as "
        + str(reply["question_type"])
        + " about the application "
        + str(reply["application_reference"])
        + " and the substation "
        + str(reply["substation_reference"])
        + (" and refused" if reply["was_refused"] else "")
        + ". The standard answers it, so the question is answered from the documents, names no "
        "application or substation and is not refused."
    )
    folder = {name for name in os.listdir(config.DOCS_DIRECTORY) if name.lower().endswith(".pdf")}
    cited = reply["source_documents"]
    searched = search_documents(knowledge_collection["collection"],
                                [chat_replies["standard_question"]])
    grounding = (searched[0][0].get("document_name")
                 if isinstance(searched, list) and len(searched) > 0
                 and isinstance(searched[0], list) and len(searched[0]) > 0
                 and isinstance(searched[0][0], dict) else None)
    assert len(cited) > 0 and set(cited) <= folder and grounding in cited, (
        "The answer cites "
        + str(cited)
        + ". An answer names the documents it came from, each of them in the document folder, "
        "and among them the one holding the passage search_documents finds closest to the "
        "question, "
        + repr(grounding)
        + "; an answer that cannot is an answer from general knowledge."
    )
    assert not set(cited) & set(config.CHAT_EXCLUDED_DOCUMENTS), (
        "The answer cites "
        + str(sorted(set(cited) & set(config.CHAT_EXCLUDED_DOCUMENTS)))
        + ", which the assistant never quotes: it records past reviews of named applications, "
        "not what the standards say."
    )
    assert len(reply["answer"].strip()) > 0 and reply["answer"] != config.CHAT_REFUSAL_MESSAGE, (
        "The answer given was "
        + repr(reply["answer"][:120])
        + ", which answers nothing."
    )
    turns = chat_replies["turns"]["body"]
    missing = sorted({"turn_id", "username", "user_role", "question_text", "question_type",
                      "application_reference", "substation_reference", "technology",
                      "answer_text", "source_documents", "closest_distance", "was_refused"}
                     - set(turns[0] if len(turns) > 0 else {}))
    assert len(turns) > 0 and len(missing) == 0, (
        "The Connections Coordinator's chat turns read back as "
        + str(len(turns))
        + " turns"
        + ("" if len(turns) == 0 else ", the first without " + str(missing))
        + ", where each turn read back carries its identifier, who asked and in which role, "
        "the question, its type, its application, substation and technology, the answer, the "
        "documents it cites, its closest distance and whether it was refused."
    )
    asked = [turn for turn in turns if turn["question_text"] == chat_replies["standard_question"]
             and turn["turn_id"] > chat_replies["latest_turn_before"]]
    assert len(asked) > 0, (
        "The Connections Coordinator's latest turns read back as "
        + str([turn["question_text"] for turn in turns])
        + ", which do not include the question just answered, so the audit trail and the chat "
        "page lose it."
    )
    latest = asked[0]
    assert (latest["question_type"] == reply["question_type"] and latest["was_refused"] is False
            and latest["answer_text"] == reply["answer"]
            and latest["application_reference"] is None
            and latest["username"] == chat_replies["asker"]
            and latest["user_role"] == config.ROLE_CONNECTIONS_COORDINATOR), (
        "The stored turn is "
        + str({key: latest[key] for key in ["username", "user_role", "question_type",
                                            "application_reference", "was_refused"]})
        + ", which is not who asked or how the question was just answered."
    )
    assert latest["closest_distance"] <= config.CHAT_MAX_RELEVANCE_DISTANCE, (
        "An answered question is stored with its closest passage "
        + str(latest["closest_distance"])
        + " away, beyond the "
        + str(config.CHAT_MAX_RELEVANCE_DISTANCE)
        + " within which an answer counts as found in the documents."
    )
    assert all(name in latest["source_documents"] for name in cited), (
        "The stored turn cites "
        + repr(latest["source_documents"][:300])
        + " where the answer cited "
        + str(cited)
        + ". The audit trail shows where each answer came from."
    )
    follow_up = "And what documentation does it ask the installer to provide?"
    with session_factory() as session:
        remembered = recall_conversation(session, chat_replies["asker"], follow_up)
        someone_else = recall_conversation(session, chat_replies["someone_else"], follow_up)
    assert all(isinstance(memory, dict)
               and isinstance(memory.get("history_text"), str)
               and isinstance(memory.get("search_text"), str)
               for memory in [remembered, someone_else]), (
        "recall_conversation returned "
        + str(remembered)[:200]
        + ", where it returns the history_text the next prompt is told and the search_text "
        "searched for it."
    )
    assert (chat_replies["standard_question"] in remembered["history_text"]
            and reply["answer"] in remembered["history_text"]), (
        "The conversation recalled for the Connections Coordinator's next question does not "
        "hold the question just asked and its answer, so a follow-up such as "
        + repr(follow_up)
        + " would be read with no idea what it follows."
    )
    assert (follow_up in remembered["search_text"]
            and chat_replies["standard_question"] in remembered["search_text"]), (
        "The text searched for the follow-up is "
        + repr(remembered["search_text"][:200])
        + ". It carries the new question and the question asked just before it."
    )
    assert (chat_replies["standard_question"] not in someone_else["history_text"]
            and chat_replies["standard_question"] not in someone_else["search_text"]), (
        "The conversation recalled for a Customer Service Advisor carries the Connections "
        "Coordinator's question "
        + repr(chat_replies["standard_question"])
        + ". Each user's memory is their own last exchanges."
    )
    planted_turns = [
        (config.CHAT_TYPE_CONNECTION_STANDARD, "What does the solar PV standard ask for first?",
         "The equipment is listed in the type-test register, as the solar PV standard states.",
         None, None, None),
        (config.CHAT_TYPE_APPLICATION_STATUS, "What is the status of LCT0500001?",
         "REV-PLANTED00001, application LCT0500001 (Heat pump), reviewed under "
         "Connection-Standard-Heat-Pump.pdf and opened by ravi: Approved.", "LCT0500001", None,
         None),
        (config.CHAT_TYPE_CONNECTION_RULES, "Who decides a review?",
         "A Connections Engineer who did not open the review decides it.", None, None, None),
        (config.CHAT_TYPE_SUBSTATION_FIGURE, "How many heat pumps are connected at SS-0412?",
         "Heat pumps are connected at 27 premises on substation SS-0412.", None, "SS-0412",
         config.HEAT_PUMP),
    ]
    planted_engine = create_engine("sqlite:///" + str(tmp_path / "planted_chat.db"))
    Base.metadata.create_all(planted_engine)
    with sessionmaker(bind=planted_engine)() as session:
        for (question_type, question_text, answer_text, application_reference,
             substation_reference, technology) in planted_turns:
            insert_chat_turn(session, {
                "username": chat_replies["asker"], "user_role": config.ROLE_CONNECTIONS_COORDINATOR,
                "question_text": question_text, "question_type": question_type,
                "application_reference": application_reference,
                "substation_reference": substation_reference, "technology": technology,
                "answer_text": answer_text, "source_documents": "", "closest_distance": 0.5,
                "was_refused": False})
        recalled = recall_conversation(session, chat_replies["asker"], "And at SS-0418?")
    planted_engine.dispose()
    assert isinstance(recalled, dict) and isinstance(recalled.get("history_text"), str), (
        "recall_conversation returned "
        + str(recalled)[:200]
        + " after four planted exchanges, where it returns the history_text of the last ones."
    )
    history = recalled["history_text"]
    kept = planted_turns[-config.CHAT_HISTORY_TURN_LIMIT:]
    dropped = planted_turns[:-config.CHAT_HISTORY_TURN_LIMIT]
    positions = [history.find(turn[1]) for turn in kept]
    assert (min(positions) >= 0 and positions == sorted(positions)
            and all(turn[1] not in history for turn in dropped)), (
        "After four planted exchanges, the conversation recalled is "
        + repr(history[:500])
        + ". It holds the last "
        + str(config.CHAT_HISTORY_TURN_LIMIT)
        + " exchanges, oldest first, and nothing before them."
    )
    carried = [fact for fact in ["REV-PLANTED00001", "ravi", "27 premises"] if fact in history]
    record_turns = [turn for turn in kept if turn[0] in config.CHAT_RECORD_TYPES]
    assert (len(carried) == 0
            and history.count(config.CHAT_RECORD_ANSWER_IN_HISTORY) == len(record_turns)
            and all(turn[2] in history for turn in kept if turn not in record_turns)), (
        "After a Connections Coordinator was told an application's status and a substation "
        "figure, the conversation recalled for the next question carries "
        + str(carried)
        + " and "
        + str(history.count(config.CHAT_RECORD_ANSWER_IN_HISTORY))
        + " times "
        + repr(config.CHAT_RECORD_ANSWER_IN_HISTORY)
        + ". The recalled conversation goes into the next prompt, and nothing from the records "
        "is ever put in front of the language model: an answer from the records is replaced by "
        "those words, while an answer from the documents is replayed as given."
    )


def test_advisor_hears_status_and_substation_figures(saved_run, staff, live_files,
                                                     processed_frames, tmp_path):
    import pandas as pd
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from chat.record_answer import answer_application_record, answer_substation_figure
    from config import config
    from persistence.database import session_factory
    from persistence.models import AgentRun, ApplicationReview, Base
    from tools.application_data_tools import fetch_application

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    assert saved_run["heard_at_gate"]["problem"] is None, saved_run["heard_at_gate"]["problem"]
    assert saved_run["heard_after"]["problem"] is None, saved_run["heard_after"]["problem"]
    assert live_files["problem"] is None, live_files["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    applications = live_files["applications"]
    reference = saved_run["application_reference"]
    technology = fetch_application(applications, reference)[config.TECHNOLOGY_COLUMN]
    status_lines = {status: config.CHAT_ADVISOR_STATUS_TEMPLATE.format(
        application_reference=reference,
        technology_application=config.TECHNOLOGY_APPLICATION_PHRASES[technology],
        status=config.REVIEW_STATUS_PHRASES[status])
        for status in [config.REVIEW_STATUS_AWAITING, config.REVIEW_STATUS_APPROVED]}
    at_gate = str(saved_run["heard_at_gate"]["advisor_status"])
    after = str(saved_run["heard_after"]["advisor_status"])
    assert (at_gate == status_lines[config.REVIEW_STATUS_AWAITING]
            and after.startswith(status_lines[config.REVIEW_STATUS_APPROVED] + " ")), (
        "A Customer Service Advisor asking about "
        + reference
        + " was told "
        + repr(at_gate[:300])
        + " while the review waited and "
        + repr(after[:300])
        + " after the approval. An advisor hears the application's technology and status, "
        "first "
        + repr(status_lines[config.REVIEW_STATUS_AWAITING])
        + " and nothing more, and then "
        + repr(status_lines[config.REVIEW_STATUS_APPROVED])
        + " followed by the conditions the household must know."
    )
    for moment, answer in [
            ("while the review waited", saved_run["heard_at_gate"]["advisor_prediction"]),
            ("after the approval", saved_run["heard_after"]["advisor_prediction"])]:
        assert answer == config.OUTCOME_NOT_PREDICTED_LINE, (
            "A Customer Service Advisor asking whether "
            + reference
            + " will need network work "
            + moment
            + " was told "
            + repr(str(answer)[:300])
            + " where they hear exactly "
            + repr(config.OUTCOME_NOT_PREDICTED_LINE)
            + ". Customer service is never told a prediction."
        )
    record = saved_run["final_record"]["body"][0]
    review_line = config.CHAT_REVIEW_TEMPLATE.format(
        review_id=saved_run["at_gate"]["review_id"], application_reference=reference,
        technology_label=config.TECHNOLOGY_LABELS[technology],
        standard_document=record["standard_document"],
        coordinator_id=saved_run["opener"]["username"],
        status=config.REVIEW_STATUS_LABELS[config.REVIEW_STATUS_APPROVED])
    figures_line = config.CHAT_FIGURES_TEMPLATE.format(
        network_work_likelihood=format(record["network_work_likelihood"],
                                       "." + str(config.LIKELIHOOD_DECIMALS) + "f"),
        likelihood_band=record["likelihood_band"],
        expected_working_days=record["duration_text"])
    decided_line = config.CHAT_DECIDED_TEMPLATE.format(
        decided_by=saved_run["decider"]["username"],
        decided_by_role=config.ROLE_CONNECTIONS_ENGINEER, decision=config.DECISION_APPROVE,
        decision_reason=saved_run["decision_reason"])
    waiting_line = config.CHAT_AWAITING_DECISION_TEMPLATE.format(
        recommended_decision=saved_run["at_gate"]["recommended_decision"])
    team_lines = str(saved_run["heard_after"]["team"]).split("\n")
    waiting_lines = str(saved_run["heard_at_gate"]["team"]).split("\n")
    assert (team_lines[0] == review_line and figures_line in team_lines
            and decided_line in team_lines and waiting_line in waiting_lines), (
        "The connections team asking about "
        + reference
        + " was told "
        + repr(str(saved_run["heard_at_gate"]["team"])[:300])
        + " while the review waited and "
        + repr(str(saved_run["heard_after"]["team"])[:500])
        + " after the approval. Connections Engineers and Coordinators hear, in the "
        "configuration's words, the review "
        + repr(review_line[:160])
        + " first, its figures "
        + repr(figures_line[:160])
        + ", what it recommends while it waits and the decision once taken."
    )
    with session_factory() as session:
        stored_review = session.get(ApplicationReview, saved_run["at_gate"]["review_id"])
        stored_run = session.get(AgentRun, saved_run["at_gate"]["run_id"])
        assert stored_review is not None and stored_run is not None, (
            "The review "
            + saved_run["at_gate"]["review_id"]
            + " or its run "
            + saved_run["at_gate"]["run_id"]
            + " is not in the database, where every review and run is stored."
        )
        review_values = {column.name: getattr(stored_review, column.name)
                         for column in ApplicationReview.__table__.columns}
        run_values = {column.name: getattr(stored_run, column.name)
                      for column in AgentRun.__table__.columns}
    by_technology = {}
    for application_reference in applications[config.APPLICATION_REFERENCE_COLUMN]:
        application = fetch_application(applications, application_reference)
        by_technology.setdefault(application[config.TECHNOLOGY_COLUMN], []).append(application)
    assert len(by_technology.get(config.HEAT_PUMP, [])) >= 3 and all(
            technology_name in by_technology for technology_name
            in config.CONNECTION_STANDARD_DOCUMENTS), (
        "The open applications hold the technologies "
        + str(sorted(by_technology))
        + ", where the supplied files hold every technology and several heat pumps."
    )
    battery = by_technology[config.BATTERY_STORAGE][0]
    heat_pumps = by_technology[config.HEAT_PUMP]
    solar = by_technology[config.SOLAR_PV][0]
    charge_point = by_technology[config.EV_CHARGE_POINT][0]
    vehicle_to_grid = by_technology[config.VEHICLE_TO_GRID][0]
    export_limit, setting = config.EXPORT_LIMIT_COLUMN, config.LOAD_LIMITER_SETTING_COLUMN
    approved, awaiting = config.REVIEW_STATUS_APPROVED, config.REVIEW_STATUS_AWAITING
    conditions = config.HOUSEHOLD_CONDITION_TEMPLATES
    approval = {config.EQUIPMENT_PHASES_COLUMN: 1,
                config.EARTHING_ARRANGEMENT_COLUMN: config.EARTHING_TN_S}
    rework = {"requirements_to_put_right": [
        {"requirement_name": "Smart charging",
         "requirement_sentence": "The equipment has smart charging functionality.",
         "result": config.REQUIREMENT_NOT_DOCUMENTED,
         "form_answers": {config.SMART_CHARGING_COLUMN: None},
         "blank_answers": [config.SMART_CHARGING_COLUMN]}],
        "blank_rating_answer": None,
        "documentation_to_provide": "The charge point's data sheet showing its smart charging.",
        "flagged_statements": []}
    cases = [
        (battery, approved, dict(approval, **{export_limit: 3.68}),
         " " + conditions[export_limit].format(
             equipment_name=config.EQUIPMENT_NAMES[config.BATTERY_STORAGE], value="3.68")),
        (heat_pumps[0], approved, dict(approval, **{setting: 55.0}),
         " " + conditions[setting].format(
             equipment_name=config.EQUIPMENT_NAMES[config.HEAT_PUMP], value="55")),
        (solar, approved, dict(approval, **{export_limit: 5.0, setting: 40.0}),
         " " + conditions[export_limit].format(
             equipment_name=config.EQUIPMENT_NAMES[config.SOLAR_PV], value="5") + " "
         + conditions[setting].format(equipment_name=config.EQUIPMENT_NAMES[config.SOLAR_PV],
                                      value="40")),
        (charge_point, approved, approval, " " + config.CHAT_NO_CONDITIONS_TEXT),
        (vehicle_to_grid, config.REVIEW_STATUS_RETURNED, rework, ""),
        (heat_pumps[1], config.REVIEW_STATUS_REFERRED, None, ""),
        (by_technology[config.EV_CHARGE_POINT][1], awaiting, None, ""),
        (heat_pumps[2], None, None, ""),
    ]
    planted_engine = create_engine("sqlite:///" + str(tmp_path / "planted_reviews.db"))
    Base.metadata.create_all(planted_engine)
    told = []
    with sessionmaker(bind=planted_engine)() as planted:
        for position, (picked, status, details, _conditions_told) in enumerate(cases, start=1):
            if status is None:
                continue
            picked_reference = picked[config.APPLICATION_REFERENCE_COLUMN]
            notice_details = None
            if details is not None:
                notice_details = dict(details, **{
                    config.APPLICATION_REFERENCE_COLUMN: picked_reference,
                    "standard_document": config.CONNECTION_STANDARD_DOCUMENTS[
                        picked[config.TECHNOLOGY_COLUMN]],
                    config.TECHNOLOGY_COLUMN: picked[config.TECHNOLOGY_COLUMN]})
            planted.add(AgentRun(**dict(
                run_values, run_id="planted-run-" + str(position),
                thread_id="planted-thread-" + str(position),
                application_reference=picked_reference)))
            planted.add(ApplicationReview(**dict(
                review_values, review_id="REV-PLANTED" + str(position).zfill(5),
                run_id="planted-run-" + str(position), application_reference=picked_reference,
                technology=picked[config.TECHNOLOGY_COLUMN], status=status,
                notice_details=notice_details)))
        planted.flush()
        for picked, status, _details, conditions_told in cases:
            picked_reference = picked[config.APPLICATION_REFERENCE_COLUMN]
            status_text = config.CHAT_AWAITING_REVIEW_TEXT
            if status is not None:
                status_text = config.REVIEW_STATUS_PHRASES[status]
            told.append((
                "a Customer Service Advisor asking about the "
                + config.TECHNOLOGY_NAMES[picked[config.TECHNOLOGY_COLUMN]]
                + " application "
                + picked_reference
                + " with a planted review "
                + str(status),
                answer_application_record(planted, staff["advisor"], applications,
                                          config.CHAT_TYPE_APPLICATION_STATUS, picked_reference),
                config.CHAT_ADVISOR_STATUS_TEMPLATE.format(
                    application_reference=picked_reference,
                    technology_application=config.TECHNOLOGY_APPLICATION_PHRASES[
                        picked[config.TECHNOLOGY_COLUMN]],
                    status=status_text)
                + conditions_told))
        unreviewed = heat_pumps[2][config.APPLICATION_REFERENCE_COLUMN]
        told.extend([
            ("a Customer Service Advisor naming no application",
             answer_application_record(planted, staff["advisor"], applications,
                                       config.CHAT_TYPE_APPLICATION_STATUS, None),
             config.CHAT_APPLICATION_NOT_NAMED_MESSAGE),
            ("a Customer Service Advisor naming LCT0099999, which is not an open application",
             answer_application_record(planted, staff["advisor"], applications,
                                       config.CHAT_TYPE_APPLICATION_STATUS, "LCT0099999"),
             config.CHAT_NO_APPLICATION_TEMPLATE.format(application_reference="LCT0099999")),
            ("a Customer Service Advisor asking how long " + unreviewed + " will take",
             answer_application_record(planted, staff["advisor"], applications,
                                       config.CHAT_TYPE_OUTCOME_PREDICTION, unreviewed),
             config.OUTCOME_NOT_PREDICTED_LINE),
            ("a Connections Coordinator naming " + unreviewed + ", which nobody has reviewed",
             answer_application_record(planted, staff["coordinator"], applications,
                                       config.CHAT_TYPE_OUTCOME_PREDICTION, unreviewed),
             config.CHAT_NO_REVIEW_TEMPLATE.format(application_reference=unreviewed)),
        ])
        refusal = None
        try:
            answer_application_record(planted, {"username": "visitor", "role": "Auditor"},
                                      applications, config.CHAT_TYPE_APPLICATION_STATUS,
                                      reference)
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
    planted_engine.dispose()
    for meaning, answer, expected in told:
        assert answer == expected, (
            "answer_application_record told "
            + meaning
            + " "
            + repr(str(answer)[:300])
            + " where it answers exactly "
            + repr(expected)
            + ". An advisor hears an application's technology and status, and once it is "
            "approved the export limit and the load limiting setting the household must know, "
            "or that it sets neither - never a prediction."
        )
    assert refusal is not None and "Auditor" in refusal, (
        "answer_application_record, asked by a role that is neither customer service nor the "
        "connections team, gave "
        + str(refusal)
        + ". Only those roles hear about an application from the records; any other is refused "
        "with a ValueError naming the role."
    )
    template = processed_frames[config.CLEAN_APPLICATION_HISTORY_EVAL_FILE].head(1)
    planted_rows = []
    for substation, technology_code, premises in [
            ("SS-9001", config.HEAT_PUMP, [1, 2, 3, 4, 5, 1]),
            ("SS-9002", config.HEAT_PUMP, [11, 12, 13, 14, 11, 12]),
            ("SS-9001", config.BATTERY_STORAGE, [21, 22, 23, 24, 25, 26, 27]),
            ("SS-9001", config.VEHICLE_TO_GRID, [31, 32, 33, 34, 35]),
            ("SS-9003", config.SOLAR_PV, [41, 42, 43, 44]),
            ("SS-9004", config.SOLAR_PV, [45, 46, 47])]:
        for premises_number in premises:
            row = template.copy()
            row[config.SUBSTATION_REFERENCE_COLUMN] = substation
            row[config.TECHNOLOGY_COLUMN] = technology_code
            row[config.PREMISES_REFERENCE_COLUMN] = "PRM" + str(9000000 + premises_number)
            planted_rows.append(row)
    planted_history = pd.concat(planted_rows, ignore_index=True)
    figure = config.CHAT_SUBSTATION_FIGURE_TEMPLATE
    plurals = config.TECHNOLOGY_PLURALS
    figure_cases = [
        ("heat pumps at SS-9001, six connections at five premises", "SS-9001", config.HEAT_PUMP,
         figure.format(technology_plural=plurals[config.HEAT_PUMP], premises_count=5,
                       substation_reference="SS-9001")),
        ("heat pumps at SS-9002, six connections at four premises", "SS-9002", config.HEAT_PUMP,
         config.GROUP_SIZE_LINE),
        ("batteries at SS-9001, at seven premises", "SS-9001", config.BATTERY_STORAGE,
         figure.format(technology_plural=plurals[config.BATTERY_STORAGE], premises_count=7,
                       substation_reference="SS-9001")),
        ("vehicle-to-grid charge points at SS-9001, at five premises", "SS-9001",
         config.VEHICLE_TO_GRID,
         figure.format(technology_plural=plurals[config.VEHICLE_TO_GRID], premises_count=5,
                       substation_reference="SS-9001")),
        ("solar PV at SS-9003, at four premises there and three at SS-9004", "SS-9003",
         config.SOLAR_PV, config.GROUP_SIZE_LINE),
        ("EV charge points at SS-9002, at none", "SS-9002", config.EV_CHARGE_POINT,
         config.GROUP_SIZE_LINE),
        ("no substation", None, config.HEAT_PUMP, config.CHAT_SUBSTATION_NOT_NAMED_MESSAGE),
        ("no technology", "SS-9001", None, config.CHAT_SUBSTATION_NOT_NAMED_MESSAGE),
    ]
    for meaning, substation, technology_code, expected in figure_cases:
        answer = answer_substation_figure(planted_history, substation, technology_code)
        assert answer == expected, (
            "Asked about "
            + meaning
            + ", answer_substation_figure gave "
            + repr(str(answer)[:300])
            + " where it answers exactly "
            + repr(expected)
            + ". A substation figure counts the different premises with that technology on that "
            "substation, and is given only where at least "
            + str(config.GROUP_SIZE_MINIMUM)
            + " premises make it up, for everyone who asks: below that the households could be "
            "told apart, so the answer is the group-size line."
        )
