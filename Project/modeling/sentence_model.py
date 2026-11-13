"""The sentence model: names the type of every sentence of an installer's statement."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import config
from modeling.model_evaluation import evaluate_text_classifier


def train_sentence_model(train_text, train_target, held_out_text, held_out_target):
    """TF-IDF words and word pairs into a logistic regression; fitted on training text only."""
    model = Pipeline([
        (config.PREPROCESSING_STEP_NAME, TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                                         sublinear_tf=True, min_df=1,
                                                         strip_accents="unicode")),
        (config.MODEL_STEP_NAME, LogisticRegression(max_iter=config.SENTENCE_MODEL_MAXIMUM_ITERATIONS,
                                                    C=10.0, random_state=config.RANDOM_SEED)),
    ])
    model.fit([str(text) for text in train_text], list(train_target))
    scores = evaluate_text_classifier(model, [str(text) for text in train_text], train_target,
                                      [str(text) for text in held_out_text], held_out_target)
    return model, scores
