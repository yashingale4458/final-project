"""Day 1 - Aparna - modeling/model_registry.py, modeling/model_evaluation.py, tools/screening_tools.py"""
import math
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

from config import config
from modeling import model_registry
from modeling.model_evaluation import evaluate_classifier, evaluate_regressor, evaluate_text_classifier
from tools.screening_tools import find_network_triggers, screen_application

# The worked example of the Connection Rules, section 13 (a standard that belongs to no technology).
WORKED_STANDARD = {
    "technology": "HP", "document_name": "Worked-Example.pdf",
    "requirements": [
        {"requirement_kind": "rated_current", "requirement_name": "Rated current", "requirement_sentence": "...",
         "rules": {"maximum_amps": Decimal("20")}},
        {"requirement_kind": "power_factor", "requirement_name": "Power factor", "requirement_sentence": "...",
         "rules": {"minimum_power_factor": Decimal("0.90")}},
        {"requirement_kind": "earthing_protection", "requirement_name": "Earthing protection", "requirement_sentence": "...",
         "rules": {"maximum_ohms": Decimal("200")}},
        {"requirement_kind": "export_limitation", "requirement_name": "Export limitation", "requirement_sentence": "...",
         "rules": {"maximum_amps": Decimal("12")}},
    ],
}
WORKED_APPLICATION = {
    "application_reference": "LCT0013000", "technology": "HP", "supply_phases": 1, "equipment_phases": 1,
    "rated_current_amps": 20.0, "power_factor": 0.895, "earthing_arrangement": "TT", "earth_electrode_ohms": None,
    "export_limitation_fitted": "Yes", "export_scheme_certified": "Yes", "export_limit_kw": 2.76,
    "looped_service": "No", "max_demand_after_connection_amps": 45.0, "cut_out_rating_amps": None,
    "load_limiter_fitted": None, "load_limiter_setting_amps": None,
}


def test_worked_example():
    screening = screen_application(WORKED_APPLICATION, WORKED_STANDARD)
    assert [r["result"] for r in screening["requirements"]] == ["Met", "Not met", "Not documented", "Met"]
    assert screening["requirements"][2]["blank_answers"] == ["earth_electrode_ohms"]
    assert (screening["requirements_met_count"], screening["requirements_not_met_count"],
            screening["requirements_not_documented_count"]) == (2, 1, 1)
    assert screening["assessment"] == "RETURN TO INSTALLER"
    assert screening["network_triggers"] == ["supply capacity"]
    assert screening["decisions_open"] == ["RETURN TO INSTALLER", "REFER FOR NETWORK ASSESSMENT"]


def test_nan_counts_as_blank_and_wrong_standard_refused():
    application = dict(WORKED_APPLICATION, earth_electrode_ohms=float("nan"), technology="EVCP")
    with pytest.raises(ValueError, match="does not apply"):
        screen_application(application, WORKED_STANDARD)


def test_exact_decimals_limit_is_met():
    standard = dict(WORKED_STANDARD, requirements=[dict(WORKED_STANDARD["requirements"][0], rules={"maximum_amps": Decimal("27.3")})])
    assert screen_application(dict(WORKED_APPLICATION, rated_current_amps=27.3), standard)["requirements"][0]["result"] == "Met"


def test_generation_trigger_and_fast_track():
    solar = {"application_reference": "LCT0000001", "technology": "PV", "looped_service": "No",
             "cut_out_rating_amps": 100, "max_demand_after_connection_amps": 30.0, "load_limiter_fitted": "No",
             "equipment_phases": 1, "inverter_capacity_kw": 5.0, "existing_generation_kw": 0.0,
             "export_limitation_fitted": "Yes", "export_scheme_certified": "Yes", "export_limit_kw": 3.68}
    assert find_network_triggers(solar) == []                                     # 21.7 A, cleared by the fast track
    assert find_network_triggers(dict(solar, inverter_capacity_kw=8.0)) == ["generation above the notify limit"]
    assert find_network_triggers(dict(solar, looped_service=None)) == ["looped service"]


def test_registry_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_METRICS_FILE", str(tmp_path / "out" / "metrics.json"))
    model = Pipeline([("preprocessing", TfidfVectorizer()), ("estimator", LogisticRegression())]).fit(["apple pie", "cherry tart"], ["x", "y"])
    model_registry.save_model(model, str(tmp_path / "models" / "m.pkl"))
    assert list(model_registry.load_model(str(tmp_path / "models" / "m.pkl")).predict(["apple pie"])) == ["x"]
    model_registry.save_feature_schema(["b", "a"], str(tmp_path / "models" / "s.json"))
    assert model_registry.load_feature_schema(str(tmp_path / "models" / "s.json")) == ["b", "a"]
    model_registry.save_metrics({"sentence": {"x": 1}})


def test_evaluation_keys():
    x = pd.DataFrame({"f": np.arange(40.0)})
    y = pd.Series((np.arange(40) % 2 == 0).astype(int) ^ (np.arange(40) > 20).astype(int))
    classifier = Pipeline([("estimator", LogisticRegression())]).fit(x, y)
    assert set(evaluate_classifier(classifier, x, y, x, y)) == {"train_roc_auc", "held_out_roc_auc", "brier_score",
        "mean_predicted_probability", "positive_share", "generalisation_gap"}
    regressor = Pipeline([("estimator", LinearRegression())]).fit(x, x["f"] * 2)
    assert evaluate_regressor(regressor, x, x["f"] * 2, x, x["f"] * 2)["held_out_r2"] == 1.0
    text = pd.Series(["red apple", "green pear", "red cherry", "green lime"])
    labels = pd.Series(["red", "green", "red", "green"])
    text_model = Pipeline([("preprocessing", TfidfVectorizer()), ("estimator", LogisticRegression())]).fit(text, labels)
    scores = evaluate_text_classifier(text_model, text, labels, text, labels)
    assert set(scores["per_class_f1"]) == {"red", "green"} and not math.isnan(scores["accuracy"])
