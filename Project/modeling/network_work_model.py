"""The network work model: the probability an application needs work on the network.

A logistic regression is used on purpose: its score moves the same way for every
application when one input changes, so turning a requirement from Met to Not met,
or a service from its own to looped, raises the likelihood across the board.
"""

from sklearn.linear_model import LogisticRegression

from config import config
from data_pipeline.feature_builder import build_model_pipeline
from modeling.model_evaluation import evaluate_classifier


def train_network_work_model(train_features, train_target, held_out_features, held_out_target):
    """Fit on the training applications only; return the pipeline and its scores."""
    estimator = LogisticRegression(max_iter=config.NETWORK_WORK_MODEL_MAXIMUM_ITERATIONS,
                                   random_state=config.RANDOM_SEED)
    model = build_model_pipeline(estimator, config.NETWORK_WORK_NUMERIC_FEATURES,
                                 config.NETWORK_WORK_CATEGORICAL_FEATURES)
    model.fit(train_features, train_target.astype(int))
    scores = evaluate_classifier(model, train_features, train_target.astype(int),
                                 held_out_features, held_out_target.astype(int))
    return model, scores
