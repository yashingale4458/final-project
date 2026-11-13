import os
from decimal import Decimal

RAW_APPLICATION_HISTORY_TRAIN_FILE = "data/application_history_train.csv"
RAW_APPLICATION_HISTORY_EVAL_FILE = "data/application_history_eval.csv"
RAW_CONNECTION_LOGS_TRAIN_FILE = "data/connection_logs_train.csv"
RAW_CONNECTION_LOGS_EVAL_FILE = "data/connection_logs_eval.csv"
RAW_STATEMENT_SENTENCES_FILE = "data/statement_sentences.csv"

CLEAN_APPLICATION_HISTORY_TRAIN_FILE = "processed/application_history_train_clean.csv"
CLEAN_APPLICATION_HISTORY_EVAL_FILE = "processed/application_history_eval_clean.csv"
CLEAN_CONNECTION_LOGS_TRAIN_FILE = "processed/connection_logs_train_clean.csv"
CLEAN_CONNECTION_LOGS_EVAL_FILE = "processed/connection_logs_eval_clean.csv"
CLEAN_STATEMENT_SENTENCES_FILE = "processed/statement_sentences_clean.csv"

DATA_QUALITY_REPORT_FILE = "output/data_quality_report.json"
JSON_INDENT = 2
TARGET_SHARE_DECIMALS = 4
DATASET_NAME_WIDTH = 30
ROW_COUNT_WIDTH = 6


NULL_TOKENS = ["", "n/a", "na", "-", "tbc", "unknown", "not recorded"]


WATTCROFT = "Wattcroft"
INSTALLERLINE = "InstallerLine"
SITEWREN = "SiteWren"

SOURCE_PORTAL_MAPPING = {
    "wattcroft": WATTCROFT,
    "installerline": INSTALLERLINE,
    "sitewren": SITEWREN,
}


BOOLEAN_YES = "Yes"
BOOLEAN_NO = "No"

BOOLEAN_MAPPING = {
    "yes": BOOLEAN_YES,
    "y": BOOLEAN_YES,
    "true": BOOLEAN_YES,
    "1": BOOLEAN_YES,
    "no": BOOLEAN_NO,
    "n": BOOLEAN_NO,
    "false": BOOLEAN_NO,
    "0": BOOLEAN_NO,
}


EV_CHARGE_POINT = "EVCP"
HEAT_PUMP = "HP"
VEHICLE_TO_GRID = "V2G"
SOLAR_PV = "PV"
BATTERY_STORAGE = "BESS"

TECHNOLOGY_MAPPING = {
    "ev charge point": EV_CHARGE_POINT,
    "ev charger": EV_CHARGE_POINT,
    "evcp": EV_CHARGE_POINT,
    "heat pump": HEAT_PUMP,
    "hp": HEAT_PUMP,
    "vehicle-to-grid charge point": VEHICLE_TO_GRID,
    "v2g charger": VEHICLE_TO_GRID,
    "v2g": VEHICLE_TO_GRID,
    "solar pv": SOLAR_PV,
    "solar panels": SOLAR_PV,
    "pv": SOLAR_PV,
    "battery storage": BATTERY_STORAGE,
    "home battery": BATTERY_STORAGE,
    "bess": BATTERY_STORAGE,
}

URGENCY_STANDARD = "Standard"
URGENCY_HEATING_FAILED = "Heating failed"

URGENCY_MAPPING = {
    "standard": URGENCY_STANDARD,
    "routine": URGENCY_STANDARD,
    "std": URGENCY_STANDARD,
    "heating failed": URGENCY_HEATING_FAILED,
    "no heating": URGENCY_HEATING_FAILED,
    "htg-fail": URGENCY_HEATING_FAILED,
}

SINGLE_PHASE = 1
THREE_PHASE = 3

PHASE_MAPPING = {
    "single": SINGLE_PHASE,
    "1ph": SINGLE_PHASE,
    "1-phase": SINGLE_PHASE,
    "1": SINGLE_PHASE,
    "three": THREE_PHASE,
    "3ph": THREE_PHASE,
    "3-phase": THREE_PHASE,
    "3": THREE_PHASE,
}

EARTHING_TN_C_S = "TN-C-S"
EARTHING_TN_S = "TN-S"
EARTHING_TT = "TT"

EARTHING_ARRANGEMENT_MAPPING = {
    "tn-c-s": EARTHING_TN_C_S,
    "pme": EARTHING_TN_C_S,
    "tncs": EARTHING_TN_C_S,
    "tn-s": EARTHING_TN_S,
    "sne": EARTHING_TN_S,
    "tns": EARTHING_TN_S,
    "tt": EARTHING_TT,
}

DETACHED_HOUSE = "Detached house"
SEMI_DETACHED_HOUSE = "Semi-detached house"
TERRACED_HOUSE = "Terraced house"
BUNGALOW = "Bungalow"
FARMHOUSE = "Farmhouse"

PREMISES_TYPE_MAPPING = {
    "detached house": DETACHED_HOUSE,
    "detached": DETACHED_HOUSE,
    "det": DETACHED_HOUSE,
    "semi-detached house": SEMI_DETACHED_HOUSE,
    "semi": SEMI_DETACHED_HOUSE,
    "terraced house": TERRACED_HOUSE,
    "terrace": TERRACED_HOUSE,
    "terr": TERRACED_HOUSE,
    "bungalow": BUNGALOW,
    "bung": BUNGALOW,
    "farmhouse": FARMHOUSE,
    "farm": FARMHOUSE,
}

SERVICE_UNDERGROUND = "Underground"
SERVICE_OVERHEAD = "Overhead"

SERVICE_TYPE_MAPPING = {
    "underground": SERVICE_UNDERGROUND,
    "ug": SERVICE_UNDERGROUND,
    "u/g": SERVICE_UNDERGROUND,
    "overhead": SERVICE_OVERHEAD,
    "oh": SERVICE_OVERHEAD,
    "o/h": SERVICE_OVERHEAD,
}

REGION_URBAN = "Urban"
REGION_SUBURBAN = "Suburban"
REGION_RURAL = "Rural"

REGION_TYPE_MAPPING = {
    "urban": REGION_URBAN,
    "urban area": REGION_URBAN,
    "urb": REGION_URBAN,
    "suburban": REGION_SUBURBAN,
    "suburban area": REGION_SUBURBAN,
    "sub": REGION_SUBURBAN,
    "rural": REGION_RURAL,
    "rural area": REGION_RURAL,
    "rur": REGION_RURAL,
}

OUTCOME_CONNECTED_AS_APPLIED = "Connected as applied"
OUTCOME_CONNECTED_AFTER_NETWORK_WORK = "Connected after network work"
OUTCOME_CONNECTED_AFTER_REWORK = "Connected after installer rework"

CONNECTION_OUTCOME_MAPPING = {
    "connected as applied": OUTCOME_CONNECTED_AS_APPLIED,
    "connected": OUTCOME_CONNECTED_AS_APPLIED,
    "conn": OUTCOME_CONNECTED_AS_APPLIED,
    "connected after network work": OUTCOME_CONNECTED_AFTER_NETWORK_WORK,
    "connected - network work": OUTCOME_CONNECTED_AFTER_NETWORK_WORK,
    "conn-nw": OUTCOME_CONNECTED_AFTER_NETWORK_WORK,
    "connected after installer rework": OUTCOME_CONNECTED_AFTER_REWORK,
    "connected - rework": OUTCOME_CONNECTED_AFTER_REWORK,
    "conn-rw": OUTCOME_CONNECTED_AFTER_REWORK,
}

WORK_UNLOOPING = "Unlooping"
WORK_CUT_OUT_UPGRADE = "Cut-out upgrade"
WORK_SERVICE_UPGRADE = "Service upgrade"
WORK_NETWORK_REINFORCEMENT = "Network reinforcement"

NETWORK_WORK_TYPE_MAPPING = {
    "unlooping": WORK_UNLOOPING,
    "unloop": WORK_UNLOOPING,
    "cut-out upgrade": WORK_CUT_OUT_UPGRADE,
    "cut-out change": WORK_CUT_OUT_UPGRADE,
    "cutout": WORK_CUT_OUT_UPGRADE,
    "service upgrade": WORK_SERVICE_UPGRADE,
    "new service": WORK_SERVICE_UPGRADE,
    "service": WORK_SERVICE_UPGRADE,
    "network reinforcement": WORK_NETWORK_REINFORCEMENT,
    "reinforcement": WORK_NETWORK_REINFORCEMENT,
    "reinf": WORK_NETWORK_REINFORCEMENT,
}

COMPLETION_COMPLETED = "Completed"
COMPLETION_AFTER_REVISIT = "Completed after revisit"
COMPLETION_AFTER_NETWORK_WORK = "Completed after network work"

COMPLETION_CODE_MAPPING = {
    "completed": COMPLETION_COMPLETED,
    "done": COMPLETION_COMPLETED,
    "comp": COMPLETION_COMPLETED,
    "completed after revisit": COMPLETION_AFTER_REVISIT,
    "done - revisit": COMPLETION_AFTER_REVISIT,
    "comp-rv": COMPLETION_AFTER_REVISIT,
    "completed after network work": COMPLETION_AFTER_NETWORK_WORK,
    "done - network": COMPLETION_AFTER_NETWORK_WORK,
    "comp-nw": COMPLETION_AFTER_NETWORK_WORK,
}

DESIGN_STATEMENT = "Design statement"
LOAD_ASSESSMENT = "Load assessment"
WIRING_NOTES = "Wiring notes"
COMMISSIONING_PLAN = "Commissioning plan"

DOCUMENT_TYPE_MAPPING = {
    "design statement": DESIGN_STATEMENT,
    "load assessment": LOAD_ASSESSMENT,
    "wiring notes": WIRING_NOTES,
    "commissioning plan": COMMISSIONING_PLAN,
}

EQUIPMENT_RATING = "Equipment rating"
EQUIPMENT_REGISTRATION = "Equipment registration"
SUPPLY_ARRANGEMENT = "Supply arrangement"
EARTHING = "Earthing"
PROTECTION_SETTINGS = "Protection settings"
EXPORT_LIMITATION = "Export limitation"
LOAD_MANAGEMENT = "Load management"
ISOLATION_AND_SWITCHING = "Isolation and switching"
BACKGROUND = "Background"

SENTENCE_TYPE_MAPPING = {
    "equipment rating": EQUIPMENT_RATING,
    "rating": EQUIPMENT_RATING,
    "equipment-rating": EQUIPMENT_RATING,
    "equipment registration": EQUIPMENT_REGISTRATION,
    "registration": EQUIPMENT_REGISTRATION,
    "equipment-registration": EQUIPMENT_REGISTRATION,
    "supply arrangement": SUPPLY_ARRANGEMENT,
    "supply": SUPPLY_ARRANGEMENT,
    "supply-arrangement": SUPPLY_ARRANGEMENT,
    "earthing": EARTHING,
    "earth": EARTHING,
    "protection settings": PROTECTION_SETTINGS,
    "protection": PROTECTION_SETTINGS,
    "protection-settings": PROTECTION_SETTINGS,
    "export limitation": EXPORT_LIMITATION,
    "export limit": EXPORT_LIMITATION,
    "export-limitation": EXPORT_LIMITATION,
    "load management": LOAD_MANAGEMENT,
    "load mgmt": LOAD_MANAGEMENT,
    "load-management": LOAD_MANAGEMENT,
    "isolation and switching": ISOLATION_AND_SWITCHING,
    "isolation & switching": ISOLATION_AND_SWITCHING,
    "isolation": ISOLATION_AND_SWITCHING,
    "isolation-and-switching": ISOLATION_AND_SWITCHING,
    "background": BACKGROUND,
    "context": BACKGROUND,
    "general": BACKGROUND,
}


APPLICATION_REFERENCE_PATTERN = r"^(?:LCT(?=[0-9]{7}$)|LC-)?([0-9]{1,7})$"
APPLICATION_REFERENCE_PREFIX = "LCT"
APPLICATION_REFERENCE_DIGITS = 7
CANONICAL_APPLICATION_REFERENCE_PATTERN = r"^LCT[0-9]{7}$"
INSTALLER_REFERENCE_PATTERN = r"^INS-[0-9]{4}$"
PREMISES_REFERENCE_PATTERN = r"^PRM[0-9]{7}$"
SUBSTATION_REFERENCE_PATTERN = r"^SS-[0-9]{4}$"

PLAIN_NUMBER_PATTERN = r"^-?[0-9]+(?:\.[0-9]+)?$"
WHOLE_NUMBER_PATTERN = r"^([0-9]+)$"
POSITIVE_WHOLE_NUMBER_PATTERN = r"^([1-9][0-9]*)$"
CUT_OUT_RATING_PATTERN = r"^([0-9]+)(?:A| amp)?$"
WORKING_DAYS_PATTERN = r"^(-?[0-9]+)(?: WD| working days?)?$"
CANONICAL_PHASES_PATTERN = r"^[13]$"

THOUSANDS_SEPARATOR = ","
WHOLE_NUMBER_DTYPE = "Int64"
DECLARED_VOLTS = Decimal("230")
WATTS_PER_KILOWATT = Decimal("1000")
OHMS_PER_KILOHM = Decimal("1000")
FRACTION_PER_PERCENT = Decimal("0.01")
AMPS_STEP = Decimal("0.1")
POWER_STEP = Decimal("0.001")
RESISTANCE_STEP = Decimal("0.1")
POWER_FACTOR_STEP = Decimal("0.0001")
AMPS_UNIT = "amps"
OHM_SIGN = "\u03a9"

KILOWATTS_NOTATION = r"^(-?[0-9]+(?:\.[0-9]+)?) ?(?:kW|kVA)$"
WATTS_NOTATION = r"^(-?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)) ?W$"
CURRENT_NOTATIONS = {
    AMPS_UNIT: r"^(-?[0-9]+(?:\.[0-9]+)?)(?:A| amps)?$",
    "kilowatts": KILOWATTS_NOTATION,
    "watts": WATTS_NOTATION,
}
POWER_UNITS = {"kilowatts": Decimal(1), "watts": Decimal(1) / WATTS_PER_KILOWATT}

POWER_READING = {
    "notations": {"kilowatts": KILOWATTS_NOTATION, "watts": WATTS_NOTATION},
    "units": POWER_UNITS,
    "step": POWER_STEP,
}

RESISTANCE_READING = {
    "notations": {"ohms": rf"^(-?[0-9]+(?:\.[0-9]+)?)(?:{OHM_SIGN}| ohms)?$",
                  "kilohms": rf"^(-?[0-9]+(?:\.[0-9]+)?)k{OHM_SIGN}$"},
    "units": {"ohms": Decimal(1), "kilohms": OHMS_PER_KILOHM},
    "step": RESISTANCE_STEP,
}

POWER_FACTOR_READING = {
    "notations": {"decimal": r"^(-?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))$",
                  "percentage": r"^(-?[0-9]+(?:\.[0-9]+)?)%$"},
    "units": {"decimal": Decimal(1), "percentage": FRACTION_PER_PERCENT},
    "step": POWER_FACTOR_STEP,
}

CURRENT_MINIMUM = 0
POWER_MINIMUM = 0
RESISTANCE_MINIMUM = float(RESISTANCE_STEP)
WORKING_DAYS_MINIMUM = 0
POWER_FACTOR_MAXIMUM = 1
CUT_OUT_SIZES = [40, 45, 60, 80, 100]


MARKUP_PATTERN = r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>"
ENCODING_ARTIFACT_PATTERN = r"&[#a-z0-9]+;|\xa0"
WHITESPACE_RUN_PATTERN = r"[\s\xa0]+"
MINIMUM_SENTENCE_WORDS = 3


APPLICATION_REFERENCE_COLUMN = "application_reference"
REVISION_NUMBER_COLUMN = "revision_number"
SUPPLY_PHASES_COLUMN = "supply_phases"
EQUIPMENT_PHASES_COLUMN = "equipment_phases"
PHASE_COLUMNS = [SUPPLY_PHASES_COLUMN, EQUIPMENT_PHASES_COLUMN]

FORM_VALUE_MAPPINGS = {
    "source_portal": SOURCE_PORTAL_MAPPING,
    "region_type": REGION_TYPE_MAPPING,
    "premises_type": PREMISES_TYPE_MAPPING,
    "technology": TECHNOLOGY_MAPPING,
    "urgency": URGENCY_MAPPING,
    "service_type": SERVICE_TYPE_MAPPING,
    "looped_service": BOOLEAN_MAPPING,
    "earthing_arrangement": EARTHING_ARRANGEMENT_MAPPING,
    "load_limiter_fitted": BOOLEAN_MAPPING,
    "export_limitation_fitted": BOOLEAN_MAPPING,
    "export_scheme_certified": BOOLEAN_MAPPING,
    "register_listed": BOOLEAN_MAPPING,
    "smart_charging": BOOLEAN_MAPPING,
    "single_controller": BOOLEAN_MAPPING,
    "open_pen_protection": BOOLEAN_MAPPING,
    "loss_of_mains_set": BOOLEAN_MAPPING,
    "isolator_fitted": BOOLEAN_MAPPING,
}

FORM_IDENTIFIER_PATTERNS = {
    "installer_reference": INSTALLER_REFERENCE_PATTERN,
    "premises_reference": PREMISES_REFERENCE_PATTERN,
    "substation_reference": SUBSTATION_REFERENCE_PATTERN,
}

FORM_WHOLE_NUMBER_PATTERNS = {
    "revision_number": POSITIVE_WHOLE_NUMBER_PATTERN,
    "cut_out_rating_amps": CUT_OUT_RATING_PATTERN,
    "existing_lct_count": WHOLE_NUMBER_PATTERN,
    "installer_planned_working_days": WORKING_DAYS_PATTERN,
}

FORM_CURRENT_PHASES = {
    "max_demand_after_connection_amps": SUPPLY_PHASES_COLUMN,
    "load_limiter_setting_amps": SUPPLY_PHASES_COLUMN,
    "rated_current_amps": EQUIPMENT_PHASES_COLUMN,
    "charging_current_amps": EQUIPMENT_PHASES_COLUMN,
}

FORM_AMOUNT_READINGS = {
    "inverter_capacity_kw": POWER_READING,
    "existing_generation_kw": POWER_READING,
    "export_limit_kw": POWER_READING,
    "power_factor": POWER_FACTOR_READING,
    "earth_electrode_ohms": RESISTANCE_READING,
}

FORM_MINIMUM_VALUES = {
    "max_demand_after_connection_amps": CURRENT_MINIMUM,
    "load_limiter_setting_amps": CURRENT_MINIMUM,
    "rated_current_amps": CURRENT_MINIMUM,
    "charging_current_amps": CURRENT_MINIMUM,
    "inverter_capacity_kw": POWER_MINIMUM,
    "existing_generation_kw": POWER_MINIMUM,
    "export_limit_kw": POWER_MINIMUM,
    "earth_electrode_ohms": RESISTANCE_MINIMUM,
    "installer_planned_working_days": WORKING_DAYS_MINIMUM,
}

FORM_MAXIMUM_VALUES = {
    "power_factor": POWER_FACTOR_MAXIMUM,
}

FORM_ALLOWED_SIZES = {
    "cut_out_rating_amps": CUT_OUT_SIZES,
}


HISTORY_TARGET_COLUMN = "connection_outcome"

HISTORY_VALUE_MAPPINGS = {
    **FORM_VALUE_MAPPINGS,
    "connection_outcome": CONNECTION_OUTCOME_MAPPING,
    "network_work_type": NETWORK_WORK_TYPE_MAPPING,
}


LOG_TARGET_COLUMN = "actual_working_days"

LOG_VALUE_MAPPINGS = {
    "source_portal": SOURCE_PORTAL_MAPPING,
    "technology": TECHNOLOGY_MAPPING,
    "premises_type": PREMISES_TYPE_MAPPING,
    "region_type": REGION_TYPE_MAPPING,
    "service_type": SERVICE_TYPE_MAPPING,
    "looped_service": BOOLEAN_MAPPING,
    "load_limiter_fitted": BOOLEAN_MAPPING,
    "export_limitation_fitted": BOOLEAN_MAPPING,
    "export_scheme_certified": BOOLEAN_MAPPING,
    "urgency": URGENCY_MAPPING,
    "network_work_needed": BOOLEAN_MAPPING,
    "completion_code": COMPLETION_CODE_MAPPING,
}

LOG_IDENTIFIER_PATTERNS = {
    "substation_reference": SUBSTATION_REFERENCE_PATTERN,
}

LOG_WHOLE_NUMBER_PATTERNS = {
    "cut_out_rating_amps": CUT_OUT_RATING_PATTERN,
    "existing_lct_count": WHOLE_NUMBER_PATTERN,
    "installer_planned_working_days": WORKING_DAYS_PATTERN,
    "actual_working_days": WORKING_DAYS_PATTERN,
}

LOG_CURRENT_PHASES = {
    "max_demand_after_connection_amps": SUPPLY_PHASES_COLUMN,
    "load_limiter_setting_amps": SUPPLY_PHASES_COLUMN,
    "rated_current_amps": EQUIPMENT_PHASES_COLUMN,
}

LOG_AMOUNT_READINGS = {
    "inverter_capacity_kw": POWER_READING,
    "existing_generation_kw": POWER_READING,
    "export_limit_kw": POWER_READING,
}

LOG_MINIMUM_VALUES = {
    "max_demand_after_connection_amps": CURRENT_MINIMUM,
    "load_limiter_setting_amps": CURRENT_MINIMUM,
    "rated_current_amps": CURRENT_MINIMUM,
    "inverter_capacity_kw": POWER_MINIMUM,
    "existing_generation_kw": POWER_MINIMUM,
    "export_limit_kw": POWER_MINIMUM,
    "installer_planned_working_days": WORKING_DAYS_MINIMUM,
    "actual_working_days": WORKING_DAYS_MINIMUM,
}

LOG_ALLOWED_SIZES = {
    "cut_out_rating_amps": CUT_OUT_SIZES,
}


SENTENCE_TARGET_COLUMN = "sentence_type"
SENTENCE_ID_COLUMN = "sentence_id"
SENTENCE_TEXT_COLUMN = "sentence_text"
SENTENCE_LENGTH_COLUMN = "sentence_length_characters"

SENTENCE_VALUE_MAPPINGS = {
    "source_portal": SOURCE_PORTAL_MAPPING,
    "document_type": DOCUMENT_TYPE_MAPPING,
    "sentence_type": SENTENCE_TYPE_MAPPING,
}

SENTENCE_WHOLE_NUMBER_PATTERNS = {
    "sentence_number": POSITIVE_WHOLE_NUMBER_PATTERN,
}


DOCS_DIRECTORY = "data/docs"
PDF_FILE_SUFFIX = ".pdf"
DATABASE_DIRECTORY = "database"
VECTOR_STORE_DIRECTORY = f"{DATABASE_DIRECTORY}/chroma_store"
VECTOR_STORE_FILE_NAME = "chroma.sqlite3"
VECTOR_COLLECTION_NAME = "connection_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
VECTOR_DISTANCE_METRIC = "cosine"
CHUNK_SIZE_CHARACTERS = 700
CHUNK_OVERLAP_CHARACTERS = 120
DOCUMENT_HEADER_PATTERN = (r"Document\s+[A-Za-z0-9 ]+\|\s*Issue\s*[0-9]+.*?Review cycle:\s*every\s+"
                           r"[a-z]+\s+months")
SECTION_HEADING_PATTERN = r"(?m)^([0-9]{1,2})\. [A-Z]"
SENTENCE_BOUNDARY_PATTERN = r"(?<=[.:;])(?<![0-9]\.)\s+"
NO_SECTION_PASSAGES_MESSAGE = ("The knowledge index holds no passage of section {section_number} "
                               "of {document_name}")
CHAT_EXCLUDED_DOCUMENTS = ["Past-Connection-Decisions-Casebook.pdf"]
CHAT_RETRIEVAL_RESULT_COUNT = 4


CONNECTION_STANDARD_DOCUMENTS = {
    EV_CHARGE_POINT: "Connection-Standard-EV-Charge-Point.pdf",
    HEAT_PUMP: "Connection-Standard-Heat-Pump.pdf",
    VEHICLE_TO_GRID: "Connection-Standard-Vehicle-To-Grid.pdf",
    SOLAR_PV: "Connection-Standard-Solar-PV.pdf",
    BATTERY_STORAGE: "Connection-Standard-Battery-Storage.pdf",
}
NO_STANDARD_CONFIGURED_MESSAGE = "No connection standard is configured for {technology}"
STANDARD_OF_ANOTHER_TECHNOLOGY_MESSAGE = ("Application {application_reference} is for the "
                                          "technology {technology}, so {document_name} does not "
                                          "apply to it")
TECHNOLOGY_SECTION_NUMBER = 2
REQUIREMENTS_SECTION_NUMBER = 4
TECHNOLOGY_CODE_PATTERN = r"The technology's code is ([A-Z0-9]+)\."
REQUIREMENTS_SECTION_PATTERN = (r"4\. Requirements .+? in this order: (.+) A requirement is never "
                                r"met by a sentence in a statement, and is never waived\.")
REQUIREMENT_NUMBER_PATTERN = r"(?:^|(?<=\.) )([1-9]) (?=[A-Z])"
REQUIREMENT_NAME_PART = "requirement_name"
REQUIREMENT_SENTENCE_PART = "requirement_sentence"
REQUIREMENT_NAME_PATTERN = rf"(?P<{REQUIREMENT_NAME_PART}>[A-Z][a-z-]*(?: [a-z-]+)*)\. "
FIGURE_PATTERN = r"[0-9]+(?:\.[0-9]+)?"

REGISTERED_EQUIPMENT_REQUIREMENT = "registered_equipment"
SMART_CHARGING_REQUIREMENT = "smart_charging"
SINGLE_CONTROLLER_REQUIREMENT = "single_controller"
RATED_CURRENT_REQUIREMENT = "rated_current"
CHARGING_CURRENT_REQUIREMENT = "charging_current"
POWER_FACTOR_REQUIREMENT = "power_factor"
EARTHING_PROTECTION_REQUIREMENT = "earthing_protection"
LOSS_OF_MAINS_REQUIREMENT = "loss_of_mains"
ISOLATION_REQUIREMENT = "isolation"
EXPORT_LIMITATION_REQUIREMENT = "export_limitation"

MAXIMUM_AMPS_RULE = "maximum_amps"
MINIMUM_POWER_FACTOR_RULE = "minimum_power_factor"
MAXIMUM_OHMS_RULE = "maximum_ohms"

REQUIREMENT_SENTENCE_PATTERNS = {
    REGISTERED_EQUIPMENT_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"The equipment is listed in the (?:approved|type-test) register\.)"),
    SMART_CHARGING_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"The equipment has smart charging functionality\.)"),
    SINGLE_CONTROLLER_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"The equipment and every back-up or boost heater are run by a single controller\.)"),
    RATED_CURRENT_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"The rated current(?:, back-up and boost heaters included,)? is at most "
        rf"(?P<{MAXIMUM_AMPS_RULE}>{FIGURE_PATTERN}) amps per phase\.)"),
    CHARGING_CURRENT_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"The charging current from the mains is at most "
        rf"(?P<{MAXIMUM_AMPS_RULE}>{FIGURE_PATTERN}) amps per phase\.)"),
    POWER_FACTOR_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        rf"The power factor is at least (?P<{MINIMUM_POWER_FACTOR_RULE}>{FIGURE_PATTERN}) at full "
        r"load\.)"),
    EARTHING_PROTECTION_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"On a PME earth the equipment has open-PEN protection, and on a TT earth its earth "
        rf"electrode is at most (?P<{MAXIMUM_OHMS_RULE}>{FIGURE_PATTERN}) ohms\.)"),
    LOSS_OF_MAINS_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"Loss-of-mains protection is set to the type-test settings\.)"),
    ISOLATION_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"An isolator that locks off is fitted on the inverter's mains side\.)"),
    EXPORT_LIMITATION_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"An export limitation scheme, where one is fitted, is certified and limits export to at "
        rf"most (?P<{MAXIMUM_AMPS_RULE}>{FIGURE_PATTERN}) amps per phase\.)"),
}


REQUIREMENT_MET = "Met"
REQUIREMENT_NOT_MET = "Not met"
REQUIREMENT_NOT_DOCUMENTED = "Not documented"

DECISION_APPROVE = "APPROVE CONNECTION"
DECISION_RETURN = "RETURN TO INSTALLER"
DECISION_REFER = "REFER FOR NETWORK ASSESSMENT"
DECISIONS_OPEN = {
    DECISION_APPROVE: [DECISION_APPROVE, DECISION_RETURN, DECISION_REFER],
    DECISION_RETURN: [DECISION_RETURN, DECISION_REFER],
    DECISION_REFER: [DECISION_RETURN, DECISION_REFER],
}

SUPPLY_MAXIMUM_AMPS = Decimal("60")
NOTIFY_GENERATION_AMPS = Decimal("16")
FAST_TRACK_GENERATION_AMPS = Decimal("32")
LOOPED_SERVICE_TRIGGER = "looped service"
SUPPLY_CAPACITY_TRIGGER = "supply capacity"
GENERATION_TRIGGER = "generation above the notify limit"
GENERATING_TECHNOLOGIES = [SOLAR_PV, BATTERY_STORAGE, VEHICLE_TO_GRID]
UNKNOWN_TECHNOLOGY_MESSAGE = ("Application {application_reference} gives the technology "
                              "{technology}, which Merrowdale has no connection standard for")

TECHNOLOGY_COLUMN = "technology"
LOOPED_SERVICE_COLUMN = "looped_service"
CUT_OUT_RATING_COLUMN = "cut_out_rating_amps"
MAXIMUM_DEMAND_COLUMN = "max_demand_after_connection_amps"
LOAD_LIMITER_FITTED_COLUMN = "load_limiter_fitted"
LOAD_LIMITER_SETTING_COLUMN = "load_limiter_setting_amps"
EARTHING_ARRANGEMENT_COLUMN = "earthing_arrangement"
OPEN_PEN_PROTECTION_COLUMN = "open_pen_protection"
EARTH_ELECTRODE_COLUMN = "earth_electrode_ohms"
POWER_FACTOR_COLUMN = "power_factor"
RATED_CURRENT_COLUMN = "rated_current_amps"
CHARGING_CURRENT_COLUMN = "charging_current_amps"
INVERTER_CAPACITY_COLUMN = "inverter_capacity_kw"
EXISTING_GENERATION_COLUMN = "existing_generation_kw"
EXPORT_LIMITATION_FITTED_COLUMN = "export_limitation_fitted"
EXPORT_SCHEME_CERTIFIED_COLUMN = "export_scheme_certified"
EXPORT_LIMIT_COLUMN = "export_limit_kw"
REGISTER_LISTED_COLUMN = "register_listed"
SMART_CHARGING_COLUMN = "smart_charging"
SINGLE_CONTROLLER_COLUMN = "single_controller"
LOSS_OF_MAINS_SET_COLUMN = "loss_of_mains_set"
ISOLATOR_FITTED_COLUMN = "isolator_fitted"

YES_OR_NO_REQUIREMENT_ANSWERS = {
    REGISTERED_EQUIPMENT_REQUIREMENT: REGISTER_LISTED_COLUMN,
    SMART_CHARGING_REQUIREMENT: SMART_CHARGING_COLUMN,
    SINGLE_CONTROLLER_REQUIREMENT: SINGLE_CONTROLLER_COLUMN,
    LOSS_OF_MAINS_REQUIREMENT: LOSS_OF_MAINS_SET_COLUMN,
    ISOLATION_REQUIREMENT: ISOLATOR_FITTED_COLUMN,
}
CURRENT_REQUIREMENT_ANSWERS = {
    RATED_CURRENT_REQUIREMENT: RATED_CURRENT_COLUMN,
    CHARGING_CURRENT_REQUIREMENT: CHARGING_CURRENT_COLUMN,
}
RATING_COLUMNS = {
    EV_CHARGE_POINT: RATED_CURRENT_COLUMN,
    HEAT_PUMP: RATED_CURRENT_COLUMN,
    VEHICLE_TO_GRID: RATED_CURRENT_COLUMN,
    SOLAR_PV: INVERTER_CAPACITY_COLUMN,
    BATTERY_STORAGE: INVERTER_CAPACITY_COLUMN,
}
SUPPLY_TRIGGER_ANSWER_COLUMNS = [LOOPED_SERVICE_COLUMN, CUT_OUT_RATING_COLUMN,
                                 MAXIMUM_DEMAND_COLUMN, LOAD_LIMITER_FITTED_COLUMN,
                                 LOAD_LIMITER_SETTING_COLUMN]
GENERATION_TRIGGER_ANSWER_COLUMNS = [EQUIPMENT_PHASES_COLUMN, INVERTER_CAPACITY_COLUMN,
                                     EXISTING_GENERATION_COLUMN, EXPORT_LIMITATION_FITTED_COLUMN,
                                     EXPORT_SCHEME_CERTIFIED_COLUMN, EXPORT_LIMIT_COLUMN]
TRIGGER_ANSWER_COLUMNS = SUPPLY_TRIGGER_ANSWER_COLUMNS + GENERATION_TRIGGER_ANSWER_COLUMNS
SCREENING_ANSWER_COLUMNS = [EQUIPMENT_PHASES_COLUMN, EARTHING_ARRANGEMENT_COLUMN,
                            OPEN_PEN_PROTECTION_COLUMN, EARTH_ELECTRODE_COLUMN, POWER_FACTOR_COLUMN,
                            RATED_CURRENT_COLUMN, CHARGING_CURRENT_COLUMN, INVERTER_CAPACITY_COLUMN,
                            EXPORT_LIMITATION_FITTED_COLUMN, EXPORT_SCHEME_CERTIFIED_COLUMN,
                            EXPORT_LIMIT_COLUMN, REGISTER_LISTED_COLUMN, SMART_CHARGING_COLUMN,
                            SINGLE_CONTROLLER_COLUMN, LOSS_OF_MAINS_SET_COLUMN,
                            ISOLATOR_FITTED_COLUMN]

REQUIREMENTS_MET_COUNT_COLUMN = "requirements_met_count"
REQUIREMENTS_NOT_MET_COUNT_COLUMN = "requirements_not_met_count"
REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN = "requirements_not_documented_count"
REQUIREMENT_COUNT_RESULTS = {
    REQUIREMENTS_MET_COUNT_COLUMN: REQUIREMENT_MET,
    REQUIREMENTS_NOT_MET_COUNT_COLUMN: REQUIREMENT_NOT_MET,
    REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN: REQUIREMENT_NOT_DOCUMENTED,
}
LOOPED_SERVICE_TRIGGER_COLUMN = "looped_service_trigger"
SUPPLY_CAPACITY_TRIGGER_COLUMN = "supply_capacity_trigger"
GENERATION_TRIGGER_COLUMN = "generation_trigger"
NETWORK_TRIGGER_FLAG_COLUMNS = {
    LOOPED_SERVICE_TRIGGER: LOOPED_SERVICE_TRIGGER_COLUMN,
    SUPPLY_CAPACITY_TRIGGER: SUPPLY_CAPACITY_TRIGGER_COLUMN,
    GENERATION_TRIGGER: GENERATION_TRIGGER_COLUMN,
}


RANDOM_SEED = 42
SENTENCE_HELD_OUT_SHARE = 0.2
METRIC_DECIMALS = 4
MODEL_FILE_COMPRESSION = 3
PREPROCESSING_STEP_NAME = "preprocessing"
MODEL_STEP_NAME = "estimator"
NUMBER_FILL_STRATEGY = "median"
CATEGORY_FILL_STRATEGY = "most_frequent"
NETWORK_WORK_MODEL_MAXIMUM_ITERATIONS = 3000
DURATION_FOREST_TREES = 300
DURATION_FOREST_LEAF_MINIMUM = 25
DURATION_FOREST_FEATURE_SHARE = 0.7
SENTENCE_MODEL_MAXIMUM_ITERATIONS = 3000

NETWORK_WORK_MODEL_FILE = "output/models/network_work_model.pkl"
DURATION_MODEL_FILE = "output/models/duration_model.pkl"
SENTENCE_MODEL_FILE = "output/models/sentence_model.pkl"
NETWORK_WORK_SCHEMA_FILE = "output/models/network_work_features.json"
DURATION_SCHEMA_FILE = "output/models/duration_features.json"
MODEL_METRICS_FILE = "output/model_metrics.json"

NETWORK_WORK_TARGET_ENCODING = {
    OUTCOME_CONNECTED_AS_APPLIED: 0,
    OUTCOME_CONNECTED_AFTER_NETWORK_WORK: 1,
    OUTCOME_CONNECTED_AFTER_REWORK: 0,
}

NETWORK_WORK_NUMERIC_FEATURES = [
    SUPPLY_PHASES_COLUMN, EQUIPMENT_PHASES_COLUMN, CUT_OUT_RATING_COLUMN, MAXIMUM_DEMAND_COLUMN,
    LOAD_LIMITER_SETTING_COLUMN, RATED_CURRENT_COLUMN, CHARGING_CURRENT_COLUMN,
    INVERTER_CAPACITY_COLUMN, EXISTING_GENERATION_COLUMN, EXPORT_LIMIT_COLUMN, "existing_lct_count",
    REQUIREMENTS_MET_COUNT_COLUMN, REQUIREMENTS_NOT_MET_COUNT_COLUMN,
    REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN, LOOPED_SERVICE_TRIGGER_COLUMN,
    SUPPLY_CAPACITY_TRIGGER_COLUMN, GENERATION_TRIGGER_COLUMN,
]

NETWORK_WORK_CATEGORICAL_FEATURES = [
    TECHNOLOGY_COLUMN, "premises_type", "region_type", "urgency", "service_type",
    LOOPED_SERVICE_COLUMN, EARTHING_ARRANGEMENT_COLUMN, LOAD_LIMITER_FITTED_COLUMN,
    EXPORT_LIMITATION_FITTED_COLUMN, EXPORT_SCHEME_CERTIFIED_COLUMN,
]

DURATION_NUMERIC_FEATURES = [
    SUPPLY_PHASES_COLUMN, EQUIPMENT_PHASES_COLUMN, CUT_OUT_RATING_COLUMN, MAXIMUM_DEMAND_COLUMN,
    LOAD_LIMITER_SETTING_COLUMN, RATED_CURRENT_COLUMN, INVERTER_CAPACITY_COLUMN,
    EXISTING_GENERATION_COLUMN, EXPORT_LIMIT_COLUMN, "existing_lct_count",
    "installer_planned_working_days", LOOPED_SERVICE_TRIGGER_COLUMN, SUPPLY_CAPACITY_TRIGGER_COLUMN,
    GENERATION_TRIGGER_COLUMN,
]

DURATION_CATEGORICAL_FEATURES = [
    TECHNOLOGY_COLUMN, "premises_type", "region_type", "urgency", "service_type",
    LOOPED_SERVICE_COLUMN, LOAD_LIMITER_FITTED_COLUMN, EXPORT_LIMITATION_FITTED_COLUMN,
    EXPORT_SCHEME_CERTIFIED_COLUMN,
]

MINIMUM_NETWORK_WORK_ROC_AUC = 0.82
MINIMUM_DURATION_R2 = 0.75
MINIMUM_SENTENCE_MACRO_F1 = 0.94


OPEN_APPLICATIONS_FILE = "data/input/open_applications.csv"
APPLICATION_DOCUMENTS_FILE = "data/input/application_documents.csv"
SUPPORTING_DOCUMENT_COUNT_COLUMN = "supporting_document_count"
LIVE_WHOLE_NUMBER_PATTERNS = {
    **FORM_WHOLE_NUMBER_PATTERNS,
    SUPPORTING_DOCUMENT_COUNT_COLUMN: WHOLE_NUMBER_PATTERN,
}
DOCUMENT_ID_COLUMN = "document_id"
DOCUMENT_TYPE_COLUMN = "document_type"
DOCUMENT_TEXT_COLUMN = "document_text"
DOCUMENT_VALUE_MAPPINGS = {
    "source_portal": SOURCE_PORTAL_MAPPING,
    DOCUMENT_TYPE_COLUMN: DOCUMENT_TYPE_MAPPING,
}
DOCUMENT_WHOLE_NUMBER_PATTERNS = {}
UNKNOWN_APPLICATION_MESSAGE = "No open application has the reference {application_reference}"
URGENCY_COLUMN = "urgency"
NOT_GIVEN_TEXT = "not given"

STATEMENT_SENTENCE_PATTERN = r"(?<=[.!?])\s+(?=[A-Za-z0-9])"
NO_STATEMENT_TEXT_MESSAGE = "Statement {document_id} has no text"
SENTENCE_TYPE_REQUIREMENT_KINDS = {
    EQUIPMENT_RATING: [RATED_CURRENT_REQUIREMENT, CHARGING_CURRENT_REQUIREMENT,
                       POWER_FACTOR_REQUIREMENT],
    EQUIPMENT_REGISTRATION: [REGISTERED_EQUIPMENT_REQUIREMENT],
    SUPPLY_ARRANGEMENT: [],
    EARTHING: [EARTHING_PROTECTION_REQUIREMENT],
    PROTECTION_SETTINGS: [LOSS_OF_MAINS_REQUIREMENT],
    EXPORT_LIMITATION: [EXPORT_LIMITATION_REQUIREMENT],
    LOAD_MANAGEMENT: [SMART_CHARGING_REQUIREMENT, SINGLE_CONTROLLER_REQUIREMENT],
    ISOLATION_AND_SWITCHING: [ISOLATION_REQUIREMENT],
    BACKGROUND: [],
}
NO_REQUIREMENT_KINDS_MESSAGE = ("No requirements are configured for the sentence type "
                                "{sentence_type}")
STATED_NUMBER_PART = "stated_number"
STATED_UNIT_PART = "stated_unit"
STATED_FIGURE_PATTERN = (rf"(?i)(?<![0-9.,])(?P<{STATED_NUMBER_PART}>"
                         rf"(?:[0-9]{{1,3}}(?:,[0-9]{{3}})+|[0-9]+)(?:\.[0-9]+)?)"
                         rf" ?(?P<{STATED_UNIT_PART}>kVA|kW|W|amps|amp|A)\b")
STATED_CURRENT_UNITS = ["a", "amp", "amps"]
STATED_POWER_UNITS = {"w": Decimal(1) / WATTS_PER_KILOWATT, "kw": Decimal(1), "kva": Decimal(1)}
FOLLOWING_WORD_PATTERN = r"[a-z-]+"
FOLLOWING_WORDS_CHECKED = 2
UNREAD_FIGURE_WORDS = ["fuse", "fuses", "cut-out", "cut-outs", "supply", "supplies", "cable",
                       "cables", "circuit", "circuits", "breaker", "breakers", "heat", "heats",
                       "thermal", "thermals"]
FORM_CURRENT_RATING_COLUMNS = [RATED_CURRENT_COLUMN, CHARGING_CURRENT_COLUMN]
FORM_POWER_RATING_COLUMNS = [INVERTER_CAPACITY_COLUMN]
DISAGREES_WITH_FORM = "DISAGREES WITH FORM"
DESCRIBES_BLANK_ANSWER = "DESCRIBES BLANK ANSWER"

SCREENING_OF_ANOTHER_APPLICATION_MESSAGE = ("The screening of {screening_reference} cannot be used "
                                            "for application {application_reference}")
LIKELIHOOD_STEP = Decimal("0.01")
LIKELIHOOD_DECIMALS = 2
AMPS_DECIMALS = 1
HIGH_LIKELIHOOD_MINIMUM = Decimal("0.50")
MEDIUM_LIKELIHOOD_MINIMUM = Decimal("0.25")
HIGH_LIKELIHOOD_BAND = "HIGH"
MEDIUM_LIKELIHOOD_BAND = "MEDIUM"
LOW_LIKELIHOOD_BAND = "LOW"
PLANNED_WORKING_DAYS_COLUMN = "installer_planned_working_days"
WORKING_DAY_STEP = Decimal("1")
DELAY_MARGIN_WORKING_DAYS = 5
EXPECTED_DAYS_WITHOUT_PLAN_TEMPLATE = ("{expected_working_days} working days; the form gives no "
                                       "planned working days, so no delay is worked out")
DELAY_TEMPLATE = ("{expected_working_days} working days against {planned_working_days} planned: "
                  "a delay is expected")
NO_DELAY_TEMPLATE = ("{expected_working_days} working days against {planned_working_days} "
                     "planned: no delay is expected")

SENTENCE_TYPE_GUIDE_DOCUMENTS = {
    EQUIPMENT_RATING: "Guide-Equipment-Ratings.pdf",
    EQUIPMENT_REGISTRATION: "Guide-Equipment-Registration.pdf",
    SUPPLY_ARRANGEMENT: "Guide-Supply-Arrangements.pdf",
    EARTHING: "Guide-Earthing.pdf",
    PROTECTION_SETTINGS: "Guide-Protection-Settings.pdf",
    EXPORT_LIMITATION: "Guide-Export-Limitation.pdf",
    LOAD_MANAGEMENT: "Guide-Load-Management.pdf",
    ISOLATION_AND_SWITCHING: "Guide-Isolation-And-Switching.pdf",
}
NO_GUIDE_CONFIGURED_MESSAGE = "No statement guide is configured for {sentence_type}"
REVIEWER_CHECKS_SECTION_NUMBER = 3
REVIEWER_CHECKS_SECTION_TITLE = "What the reviewer checks"
DOCUMENTATION_SECTION_NUMBER = 6
DOCUMENTATION_SECTION_TITLE = "Documentation to provide"
CONNECTION_APPROVAL_SECTION_NUMBER = 7
CONNECTION_APPROVAL_SECTION_TITLE = "The connection approval"
HOUSEHOLD_NOTICE_SECTION_NUMBER = 8
HOUSEHOLD_NOTICE_SECTION_TITLE = "The household notice"
SECTION_TITLE_MISMATCH_MESSAGE = ("Section {section_number} of {document_name} is not "
                                  "\"{section_title}\"")

APPROVAL_DETAIL_COLUMNS = [TECHNOLOGY_COLUMN, EQUIPMENT_PHASES_COLUMN, EARTHING_ARRANGEMENT_COLUMN]
APPROVAL_CONDITION_COLUMNS = {
    EV_CHARGE_POINT: [LOAD_LIMITER_SETTING_COLUMN],
    HEAT_PUMP: [LOAD_LIMITER_SETTING_COLUMN],
    VEHICLE_TO_GRID: [EXPORT_LIMIT_COLUMN, LOAD_LIMITER_SETTING_COLUMN],
    SOLAR_PV: [EXPORT_LIMIT_COLUMN, LOAD_LIMITER_SETTING_COLUMN],
    BATTERY_STORAGE: [EXPORT_LIMIT_COLUMN, LOAD_LIMITER_SETTING_COLUMN],
}
REQUIREMENT_ANSWER_COLUMNS = {
    REGISTERED_EQUIPMENT_REQUIREMENT: [REGISTER_LISTED_COLUMN],
    SMART_CHARGING_REQUIREMENT: [SMART_CHARGING_COLUMN],
    SINGLE_CONTROLLER_REQUIREMENT: [SINGLE_CONTROLLER_COLUMN],
    RATED_CURRENT_REQUIREMENT: [RATED_CURRENT_COLUMN],
    CHARGING_CURRENT_REQUIREMENT: [CHARGING_CURRENT_COLUMN],
    POWER_FACTOR_REQUIREMENT: [POWER_FACTOR_COLUMN],
    EARTHING_PROTECTION_REQUIREMENT: [EARTHING_ARRANGEMENT_COLUMN, OPEN_PEN_PROTECTION_COLUMN,
                                      EARTH_ELECTRODE_COLUMN],
    LOSS_OF_MAINS_REQUIREMENT: [LOSS_OF_MAINS_SET_COLUMN],
    ISOLATION_REQUIREMENT: [ISOLATOR_FITTED_COLUMN],
    EXPORT_LIMITATION_REQUIREMENT: [EXPORT_LIMITATION_FITTED_COLUMN, EXPORT_SCHEME_CERTIFIED_COLUMN,
                                    EXPORT_LIMIT_COLUMN, EQUIPMENT_PHASES_COLUMN],
}
SUBSTATION_REFERENCE_COLUMN = "substation_reference"
ASSESSMENT_BRIEF_DETAIL_COLUMNS = [
    SUBSTATION_REFERENCE_COLUMN, SUPPLY_PHASES_COLUMN, EQUIPMENT_PHASES_COLUMN,
    LOOPED_SERVICE_COLUMN, CUT_OUT_RATING_COLUMN, MAXIMUM_DEMAND_COLUMN, LOAD_LIMITER_FITTED_COLUMN,
    LOAD_LIMITER_SETTING_COLUMN, RATED_CURRENT_COLUMN, INVERTER_CAPACITY_COLUMN,
    EXISTING_GENERATION_COLUMN, EXPORT_LIMITATION_FITTED_COLUMN, EXPORT_SCHEME_CERTIFIED_COLUMN,
    EXPORT_LIMIT_COLUMN,
]

DECISIONS = [DECISION_APPROVE, DECISION_RETURN, DECISION_REFER]
APPROVE_CLOSED_REASONS = {
    DECISION_RETURN: "a requirement is Not met or Not documented, or the rating is blank",
    DECISION_REFER: "a network trigger applies",
}
REVIEW_STATUS_APPROVED = "APPROVED"
REVIEW_STATUS_RETURNED = "RETURNED_TO_INSTALLER"
REVIEW_STATUS_REFERRED = "REFERRED_FOR_ASSESSMENT"
DECIDED_REVIEW_STATUSES = [REVIEW_STATUS_APPROVED, REVIEW_STATUS_RETURNED, REVIEW_STATUS_REFERRED]
ALREADY_DECIDED_MESSAGE = ("The review of application {application_reference} has already been "
                           "decided")
DECISION_IN_PROGRESS_MESSAGE = ("{pending_decided_by} has sent {pending_decision} for application "
                                "{application_reference} and it is still being recorded; only "
                                "{pending_decided_by} can finish it, by sending the same decision "
                                "with the same reason again")
OWN_REVIEW_MESSAGE = ("{decided_by} opened the review of application {application_reference} and "
                      "may not decide it")
UNKNOWN_DECISION_MESSAGE = ("{decision} is not one of the decisions on application "
                            "{application_reference}: {decisions}")
DECISION_NOT_OPEN_MESSAGE = ("{decision} is not open for application {application_reference}: "
                             "{reason}")
DECISION_RECORD_FIELDS = ["decision", "decided_by", "decided_by_role", "decision_reason"]
ROLE_CONNECTIONS_ENGINEER = "Connections Engineer"
DECISION_FIELDS_MISSING_MESSAGE = ("The decision on application {application_reference} gives no "
                                   "{fields}")
DECIDING_ROLE_MESSAGE = ("The role {role} may not decide application {application_reference}: "
                         "only a {engineer_role} decides a review")


ORGANISATION_NAME = "Merrowdale Power Networks"
SOURCE_PORTAL_COLUMN = "source_portal"
PREMISES_TYPE_COLUMN = "premises_type"
REGION_TYPE_COLUMN = "region_type"
SERVICE_TYPE_COLUMN = "service_type"
DOCUMENT_TITLE_COLUMN = "document_title"
TECHNOLOGY_NAMES = {
    EV_CHARGE_POINT: "EV charge point",
    HEAT_PUMP: "heat pump",
    VEHICLE_TO_GRID: "vehicle-to-grid charge point",
    SOLAR_PV: "solar PV",
    BATTERY_STORAGE: "battery storage",
}
FORM_ANSWER_LABELS = {
    TECHNOLOGY_COLUMN: "the technology",
    SUBSTATION_REFERENCE_COLUMN: "the substation",
    PREMISES_TYPE_COLUMN: "the premises type",
    REGION_TYPE_COLUMN: "the region type",
    URGENCY_COLUMN: "the urgency",
    SERVICE_TYPE_COLUMN: "the service type",
    SUPPLY_PHASES_COLUMN: "the supply's phases",
    EQUIPMENT_PHASES_COLUMN: "the equipment's phases",
    LOOPED_SERVICE_COLUMN: "whether the service is looped",
    CUT_OUT_RATING_COLUMN: "the cut-out rating, in amps",
    MAXIMUM_DEMAND_COLUMN: "the maximum demand after connection, in amps per phase",
    LOAD_LIMITER_FITTED_COLUMN: "whether a load limiting device is fitted",
    LOAD_LIMITER_SETTING_COLUMN: "the load limiting device's setting, in amps per phase",
    RATED_CURRENT_COLUMN: "the rated current, in amps per phase",
    CHARGING_CURRENT_COLUMN: "the charging current from the mains, in amps per phase",
    INVERTER_CAPACITY_COLUMN: "the inverter capacity, in kilowatts",
    EXISTING_GENERATION_COLUMN: "the existing generation at the house, in kilowatts",
    EXPORT_LIMITATION_FITTED_COLUMN: "whether an export limitation scheme is fitted",
    EXPORT_SCHEME_CERTIFIED_COLUMN: "whether the export limitation scheme is certified",
    EXPORT_LIMIT_COLUMN: "the export limit, in kilowatts",
    REGISTER_LISTED_COLUMN: "whether the equipment is listed in the register",
    SMART_CHARGING_COLUMN: "whether the equipment has smart charging",
    SINGLE_CONTROLLER_COLUMN: "whether one controller runs the equipment and its heaters",
    POWER_FACTOR_COLUMN: "the power factor at full load",
    EARTHING_ARRANGEMENT_COLUMN: "the earthing arrangement the equipment will use",
    OPEN_PEN_PROTECTION_COLUMN: "whether open-PEN protection is fitted",
    EARTH_ELECTRODE_COLUMN: "the earth electrode's resistance, in ohms",
    LOSS_OF_MAINS_SET_COLUMN: "whether loss-of-mains protection is set to the type-test settings",
    ISOLATOR_FITTED_COLUMN: "whether a lockable isolator is fitted on the inverter's mains side",
    "existing_lct_count": "the low-carbon technologies already at the house",
    PLANNED_WORKING_DAYS_COLUMN: "the installer's planned working days",
    REQUIREMENTS_MET_COUNT_COLUMN: "requirements Met",
    REQUIREMENTS_NOT_MET_COUNT_COLUMN: "requirements Not met",
    REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN: "requirements Not documented",
    LOOPED_SERVICE_TRIGGER_COLUMN: "the looped service trigger, 1 when it applies",
    SUPPLY_CAPACITY_TRIGGER_COLUMN: "the supply capacity trigger, 1 when it applies",
    GENERATION_TRIGGER_COLUMN: "the generation trigger, 1 when it applies",
}
STATEMENT_GUIDE_ANSWER_COLUMNS = [SUPPLY_PHASES_COLUMN, EQUIPMENT_PHASES_COLUMN,
                                  LOOPED_SERVICE_COLUMN, CUT_OUT_RATING_COLUMN,
                                  MAXIMUM_DEMAND_COLUMN, LOAD_LIMITER_FITTED_COLUMN,
                                  LOAD_LIMITER_SETTING_COLUMN, RATED_CURRENT_COLUMN,
                                  CHARGING_CURRENT_COLUMN, INVERTER_CAPACITY_COLUMN,
                                  EXISTING_GENERATION_COLUMN, EARTHING_ARRANGEMENT_COLUMN,
                                  EXPORT_LIMITATION_FITTED_COLUMN, EXPORT_SCHEME_CERTIFIED_COLUMN,
                                  EXPORT_LIMIT_COLUMN]
NETWORK_BLANK_ANSWER_RULE_TEXT = ("A blank answer never lets an application slip past the network: "
                                  "a blank looped service counts as looped; a blank cut-out rating "
                                  "counts as reached, and so does a blank demand that counts; a "
                                  "blank answer to whether a load limiting device is fitted makes "
                                  "the demand that counts the higher of the declared maximum "
                                  "demand and any setting given, or the declared demand alone "
                                  "where either is blank; and a blank inverter capacity, existing "
                                  "generation or equipment phases counts as above the notify "
                                  "limit, whatever the scheme.")
CURRENT_RULE_TEXT = (f"Every current is in amps per phase. A power meets a limit in amps only "
                     f"after it is made amps per phase at the declared {DECLARED_VOLTS} volts, on "
                     f"the equipment's phases for its own rating, charging current, generating "
                     f"capacity and export limit, and on the supply's phases for the maximum "
                     f"demand and a load limiting device's setting, as the screening has already "
                     f"done.")
ASSESSMENT_RULE_TEXT = (f"The assessment is {DECISION_RETURN} when any requirement is Not met or "
                        f"Not documented or the rating is blank; otherwise {DECISION_REFER} when "
                        f"any network trigger applies; otherwise {DECISION_APPROVE}.")
DECISION_DESCRIPTIONS = {
    DECISION_APPROVE: ("the equipment may be connected as applied, and a connection approval to "
                       "the installer and then a household notice to the people who live at the "
                       "premises are written"),
    DECISION_RETURN: ("the application goes back to its installer with a rework request naming "
                      "everything to put right"),
    DECISION_REFER: ("the application goes to Network Planning with its review and an assessment "
                     "brief, where an exception can be argued"),
}
DECISION_OPEN_CONDITIONS = {
    DECISION_APPROVE: (f"It is open only when the assessment is {DECISION_APPROVE}: every "
                       f"requirement Met, the rating stated and no network trigger."),
    DECISION_RETURN: "It is always open.",
    DECISION_REFER: "It is always open.",
}
BLANK_INPUT_RULE_TEXT = (f"An input shown as {NOT_GIVEN_TEXT} is filled as the model's training "
                         f"filled it: a number with the {NUMBER_FILL_STRATEGY} of the training "
                         f"data and an answer in words with its most common value.")

NOTICE_KINDS = {
    DECISION_APPROVE: "connection approval",
    DECISION_RETURN: "rework request",
}
NOTICE_DETAIL_LABELS = {
    APPLICATION_REFERENCE_COLUMN: "Application reference",
    TECHNOLOGY_COLUMN: "Technology",
    "standard_document": "Connection standard",
    EQUIPMENT_PHASES_COLUMN: "The equipment's phases",
    EARTHING_ARRANGEMENT_COLUMN: "Earthing arrangement",
    RATED_CURRENT_COLUMN: "Rated current, in amps per phase",
    INVERTER_CAPACITY_COLUMN: "Inverter capacity, in kilowatts",
    EXPORT_LIMIT_COLUMN: "Export limit, in kilowatts",
    LOAD_LIMITER_SETTING_COLUMN: "Load limiting device's setting, in amps per phase",
}
RATING_TERMS = {
    RATED_CURRENT_COLUMN: "rated current",
    INVERTER_CAPACITY_COLUMN: "inverter capacity",
}
REWORK_REQUEST_GUIDANCE = ("The rework request tells the installer every requirement that is not "
                           "met or not documented and the answers each needs, the documentation "
                           "to provide as the technology's own standard gives it, and every "
                           "statement sentence flagged against the form, to be settled by the "
                           "installer. When the rating is blank, it asks for it: every application "
                           "states its equipment's rating.")
NOTICE_SIGN_OFF = f"Connections, {ORGANISATION_NAME}"
NOTICE_FORBIDDEN_TERMS = ["likelihood", "probability", "expected working days"]
TEMPLATE_SLOT_CHARACTERS = ["[", "]"]
SIGN_OFF_SPACING_PATTERN = r"[\W_]+"
HOUSEHOLD_NOTICE_WITHHELD_PATTERNS = {
    "the substation": r"\bss-[0-9]{4}\b|\bsubstations?\b",
    "the cut-out rating": r"\bcut-?outs?\b",
    "the maximum demand": r"\bmaximum demand\b",
}
ROLE_CONNECTIONS_COORDINATOR = "Connections Coordinator"
ROLE_CUSTOMER_SERVICE_ADVISOR = "Customer Service Advisor"
SEED_USER_ACCOUNTS = [
    {"username": "amira", "display_name": "Amira Okafor", "role": ROLE_CONNECTIONS_ENGINEER,
     "password": "11111111"},
    {"username": "lewis", "display_name": "Lewis Carrick", "role": ROLE_CONNECTIONS_ENGINEER,
     "password": "11111111"},
    {"username": "megan", "display_name": "Megan Thorne", "role": ROLE_CONNECTIONS_ENGINEER,
     "password": "11111111"},
    {"username": "ravi", "display_name": "Ravi Menon", "role": ROLE_CONNECTIONS_COORDINATOR,
     "password": "22222222"},
    {"username": "hannah", "display_name": "Hannah Price", "role": ROLE_CONNECTIONS_COORDINATOR,
     "password": "22222222"},
    {"username": "callum", "display_name": "Callum Doyle", "role": ROLE_CONNECTIONS_COORDINATOR,
     "password": "22222222"},
    {"username": "nadia", "display_name": "Nadia Haddad", "role": ROLE_CONNECTIONS_COORDINATOR,
     "password": "22222222"},
    {"username": "owen", "display_name": "Owen Pritchard", "role": ROLE_CONNECTIONS_COORDINATOR,
     "password": "22222222"},
    {"username": "leah", "display_name": "Leah Simmons", "role": ROLE_CONNECTIONS_COORDINATOR,
     "password": "22222222"},
    {"username": "zara", "display_name": "Zara Bell", "role": ROLE_CUSTOMER_SERVICE_ADVISOR,
     "password": "33333333"},
    {"username": "joel", "display_name": "Joel Kerr", "role": ROLE_CUSTOMER_SERVICE_ADVISOR,
     "password": "33333333"},
    {"username": "ines", "display_name": "Ines Moreau", "role": ROLE_CUSTOMER_SERVICE_ADVISOR,
     "password": "33333333"},
    {"username": "dylan", "display_name": "Dylan Shaw", "role": ROLE_CUSTOMER_SERVICE_ADVISOR,
     "password": "33333333"},
    {"username": "maya", "display_name": "Maya Chen", "role": ROLE_CUSTOMER_SERVICE_ADVISOR,
     "password": "33333333"},
]

APPLICATION_DATABASE_FILE = f"{DATABASE_DIRECTORY}/phasewise.db"
APPLICATION_DATABASE_URL = f"sqlite:///./{APPLICATION_DATABASE_FILE}"
CHECKPOINT_DATABASE_FILE = f"{DATABASE_DIRECTORY}/checkpoint.db"
AGENT_RUNS_TABLE = "agent_runs"
APPLICATION_REVIEWS_TABLE = "application_reviews"
REVIEW_DECISIONS_TABLE = "review_decisions"
LLM_CALLS_TABLE = "llm_calls"
RUN_ID_PREFIX = "RUN-"
REVIEW_ID_PREFIX = "REV-"
IDENTIFIER_HEX_LENGTH = 12
AGENT_RUN_STATUS_RUNNING = "RUNNING"
AGENT_RUN_STATUS_AWAITING_DECISION = "AWAITING_DECISION"
AGENT_RUN_STATUS_COMPLETE = "COMPLETE"
REVIEW_STATUS_AWAITING = "AWAITING_DECISION"
REVIEW_STATUS_BY_DECISION = {
    DECISION_APPROVE: REVIEW_STATUS_APPROVED,
    DECISION_RETURN: REVIEW_STATUS_RETURNED,
    DECISION_REFER: REVIEW_STATUS_REFERRED,
}

ENV_FILE = ".env"
GEMINI_API_KEY_VARIABLE = "GEMINI_API_KEY"
GEMINI_MODEL_VARIABLE = "GEMINI_MODEL"
GEMINI_MAX_OUTPUT_TOKENS = 8192
GEMINI_TEMPERATURE = 0.2
MILLISECONDS_PER_SECOND = 1000
AGENT_NAME_PLANNER = "review_planner_agent"
AGENT_NAME_REQUIREMENTS = "requirements_analyst_agent"
AGENT_NAME_EVIDENCE = "evidence_analyst_agent"
AGENT_NAME_RISK = "risk_analyst_agent"
AGENT_NAME_WRITER = "review_writer_agent"
AGENT_NAME_INSTALLER_NOTICE = "installer_notice_writer_agent"
AGENT_NAME_HOUSEHOLD_NOTICE = "household_notice_writer_agent"
RESPONSE_DECORATION_CHARACTERS = "*#`_~ \t:"
RESPONSE_DECORATION_EDGE_PATTERN = r"^[\W_]+|[\W_]+$"


AGENT_RUN_STATUS_FAILED = "FAILED"
LLM_CALLS_BEFORE_DECISION = 5
LLM_CALLS_BY_DECISION = {
    DECISION_APPROVE: 7,
    DECISION_RETURN: 6,
    DECISION_REFER: 5,
}
REVIEW_STATUS_LABELS = {
    REVIEW_STATUS_AWAITING: "Awaiting a decision",
    REVIEW_STATUS_APPROVED: "Approved",
    REVIEW_STATUS_RETURNED: "Returned to installer",
    REVIEW_STATUS_REFERRED: "Referred for assessment",
}
REVIEW_STATUS_OPTIONS = list(REVIEW_STATUS_LABELS)
REVIEW_STATUS_PHRASES = {
    REVIEW_STATUS_AWAITING: "awaiting a decision",
    REVIEW_STATUS_APPROVED: "approved",
    REVIEW_STATUS_RETURNED: "returned to the installer",
    REVIEW_STATUS_REFERRED: "referred for a network assessment",
}
DECISION_OUTCOME_TEXTS = {
    DECISION_APPROVE: "the connection approval and the household notice are written",
    DECISION_RETURN: "the rework request is written",
    DECISION_REFER: "the assessment brief is recorded for Network Planning",
}
TECHNOLOGY_LABELS = {
    EV_CHARGE_POINT: "EV charge point",
    HEAT_PUMP: "Heat pump",
    VEHICLE_TO_GRID: "Vehicle-to-grid charge point",
    SOLAR_PV: "Solar PV",
    BATTERY_STORAGE: "Battery storage",
}
QUEUE_TECHNOLOGY_ALL = "All"
QUEUE_TECHNOLOGY_LABELS = {QUEUE_TECHNOLOGY_ALL: "All technologies", **TECHNOLOGY_LABELS}
QUEUE_TECHNOLOGY_OPTIONS = list(QUEUE_TECHNOLOGY_LABELS)
COORDINATOR_REQUEST_MINIMUM_CHARACTERS = 5
COORDINATOR_REQUEST_MAXIMUM_CHARACTERS = 1000
DECISION_REASON_MINIMUM_CHARACTERS = 10
DECISION_REASON_MAXIMUM_CHARACTERS = 1000

ROLES_SEEING_EVERY_REVIEW = [ROLE_CONNECTIONS_ENGINEER, ROLE_CONNECTIONS_COORDINATOR]
PAGE_QUEUE = "Application queue"
PAGE_DETAIL = "Application detail"
PAGE_DECISION = "Decision queue"
PAGE_AUDIT = "Audit trail"
PAGE_CHAT = "Connection chat"
ROLE_PAGE_ACCESS = {
    ROLE_CONNECTIONS_ENGINEER: [PAGE_QUEUE, PAGE_DETAIL, PAGE_DECISION, PAGE_AUDIT, PAGE_CHAT],
    ROLE_CONNECTIONS_COORDINATOR: [PAGE_QUEUE, PAGE_DETAIL, PAGE_AUDIT, PAGE_CHAT],
    ROLE_CUSTOMER_SERVICE_ADVISOR: [PAGE_CHAT],
}
AREA_QUEUE = "queue"
AREA_REVIEW = "review"
AREA_DECISION = "decision"
AREA_AUDIT = "audit"
AREA_CHAT = "chat"
ROLE_AREA_ACCESS = {
    ROLE_CONNECTIONS_ENGINEER: [AREA_QUEUE, AREA_REVIEW, AREA_DECISION, AREA_AUDIT, AREA_CHAT],
    ROLE_CONNECTIONS_COORDINATOR: [AREA_QUEUE, AREA_REVIEW, AREA_AUDIT, AREA_CHAT],
    ROLE_CUSTOMER_SERVICE_ADVISOR: [AREA_CHAT],
}
AREA_BY_ROUTE = {
    ("GET", "/queue"): AREA_QUEUE,
    ("GET", "/queue/{application_reference}"): AREA_QUEUE,
    ("POST", "/reviews"): AREA_REVIEW,
    ("GET", "/reviews"): AREA_AUDIT,
    ("POST", "/reviews/{review_id}/decision"): AREA_DECISION,
    ("GET", "/reviews/{review_id}/calls"): AREA_AUDIT,
    ("POST", "/chat/ask"): AREA_CHAT,
    ("GET", "/chat/turns"): AREA_CHAT,
    ("GET", "/audit/chat-turns"): AREA_AUDIT,
}
USERNAME_WIDTH = 8
DISPLAY_NAME_WIDTH = 16
USER_ACCOUNTS_TABLE = "user_accounts"
CHAT_TURNS_TABLE = "chat_turns"

PASSWORD_HASH_ALGORITHM = "sha256"
PASSWORD_HASH_ROUNDS = 200000
PASSWORD_SALT_BYTES = 16
SESSION_TOKEN_BYTES = 24
SESSION_TOKEN_HEADER = "X-Session-Token"
SIGN_IN_FAILED_MESSAGE = "Username or password is not correct"
SIGN_IN_REQUIRED_MESSAGE = "Sign in first"
SESSION_ENDED_MESSAGE = ("Your session ended: this account has signed in somewhere else or "
                         "signed out. Sign in again to carry on.")
AREA_REFUSED_MESSAGE = "The {role} role does not include the {area} area"
REVIEW_NOT_FOUND_MESSAGE = "No application review has that identifier"
APPLICATION_ALREADY_REVIEWED_MESSAGE = ("Application {application_reference} has already been "
                                        "reviewed; an application in the queue is reviewed once")
REVIEW_OUTSIDE_QUEUE_MESSAGE = ("Review {review_id} is of application {application_reference}, "
                                "which is not in the application queue")
CHECKPOINT_MISSING_MESSAGE = "Review {review_id} has no workflow checkpoint in {checkpoint_file}"

AGENT_NAME_CONNECTION_ASSISTANT = "connection_assistant_agent"
CHAT_TYPE_CONNECTION_STANDARD = "CONNECTION STANDARD"
CHAT_TYPE_CONNECTION_RULES = "CONNECTION RULES"
CHAT_TYPE_STATEMENT_GUIDANCE = "STATEMENT GUIDANCE"
CHAT_TYPE_APPLICATION_STATUS = "APPLICATION STATUS"
CHAT_TYPE_OUTCOME_PREDICTION = "OUTCOME PREDICTION"
CHAT_TYPE_SUBSTATION_FIGURE = "SUBSTATION FIGURE"
CHAT_TYPE_OUT_OF_SCOPE = "OUT OF SCOPE"
CHAT_TYPE_OPTIONS = [CHAT_TYPE_CONNECTION_STANDARD, CHAT_TYPE_CONNECTION_RULES,
                     CHAT_TYPE_STATEMENT_GUIDANCE, CHAT_TYPE_APPLICATION_STATUS,
                     CHAT_TYPE_OUTCOME_PREDICTION, CHAT_TYPE_SUBSTATION_FIGURE,
                     CHAT_TYPE_OUT_OF_SCOPE]
CHAT_ANSWERED_TYPES = [CHAT_TYPE_CONNECTION_STANDARD, CHAT_TYPE_CONNECTION_RULES,
                       CHAT_TYPE_STATEMENT_GUIDANCE]
CHAT_APPLICATION_TYPES = [CHAT_TYPE_APPLICATION_STATUS, CHAT_TYPE_OUTCOME_PREDICTION]
CHAT_RECORD_TYPES = [CHAT_TYPE_APPLICATION_STATUS, CHAT_TYPE_OUTCOME_PREDICTION,
                     CHAT_TYPE_SUBSTATION_FIGURE]
CHAT_QUESTION_MINIMUM_CHARACTERS = 3
CHAT_QUESTION_MAXIMUM_CHARACTERS = 1000
CHAT_MAX_RELEVANCE_DISTANCE = 0.68
CHAT_HISTORY_TURN_LIMIT = 3
CHAT_PAGE_TURN_LIMIT = 20
CHAT_AUDIT_MAX_ROWS = 200
CHAT_NO_VALUE_REPLY = "none"
CHAT_FIRST_QUESTION_TEXT = "This is the first question in the conversation."
CHAT_RECORD_ANSWER_IN_HISTORY = ("(answered from the connection records, which are not repeated "
                                 "here)")
APPLICATION_REFERENCE_IN_TEXT_PATTERN = (r"(?i)(?<![0-9A-Za-z-])(?:LCT[0-9]{7}|LC-[0-9]{1,7})"
                                         r"(?![0-9])")
NUMBER_IN_TEXT_PATTERN = r"\b{number}\b"
SUBSTATION_IN_TEXT_PATTERN = r"(?i)(?<![0-9A-Za-z-])SS-[0-9]{4}(?![0-9])"
CHAT_TECHNOLOGY_PATTERNS = {
    EV_CHARGE_POINT: (r"(?<!vehicle-to-grid )(?<!vehicle to grid )(?<!v2g )"
                      r"\b(?:ev )?(?:charge ?points?|chargers?)\b"),
    HEAT_PUMP: r"\bheat ?pumps?\b",
    VEHICLE_TO_GRID: r"\bvehicle[- ]to[- ]grid\b|\bv2g\b",
    SOLAR_PV: r"\bsolar\b",
    BATTERY_STORAGE: r"\bbatter(?:y|ies)\b",
}
CHAT_REFUSAL_MESSAGE = "The connection documents do not answer that question."
OUTCOME_NOT_PREDICTED_LINE = ("An application's outcome is decided by a Connections Engineer, and "
                              "customer service cannot predict whether network work is needed or "
                              "how long a connection will take.")
CHAT_NO_APPLICATION_TEMPLATE = ("No open application is recorded under reference "
                                "{application_reference}.")
CHAT_APPLICATION_NOT_NAMED_MESSAGE = ("Name the application by its reference, as LCT0052118, "
                                      "LC-52118 or 52118.")
CHAT_AWAITING_REVIEW_TEXT = "awaiting review"
TECHNOLOGY_APPLICATION_PHRASES = {
    EV_CHARGE_POINT: "an EV charge point application",
    HEAT_PUMP: "a heat pump application",
    VEHICLE_TO_GRID: "a vehicle-to-grid charge point application",
    SOLAR_PV: "a solar PV application",
    BATTERY_STORAGE: "a battery storage application",
}
CHAT_ADVISOR_STATUS_TEMPLATE = "{application_reference}, {technology_application}: {status}."
EQUIPMENT_NAMES = {
    EV_CHARGE_POINT: "charge point",
    HEAT_PUMP: "heat pump",
    VEHICLE_TO_GRID: "charge point",
    SOLAR_PV: "solar PV system",
    BATTERY_STORAGE: "battery",
}
HOUSEHOLD_CONDITION_TEMPLATES = {
    EXPORT_LIMIT_COLUMN: "The {equipment_name}'s export to the network is limited to {value} kW.",
    LOAD_LIMITER_SETTING_COLUMN: ("A load limiting device holds the house to {value} amps per "
                                  "phase."),
}
CHAT_NO_CONDITIONS_TEXT = "The approval sets no export limit and no load limiting setting."
CHAT_NO_REVIEW_TEMPLATE = ("No review of application {application_reference} has been recorded: "
                           "it is awaiting review.")
CHAT_REVIEW_TEMPLATE = ("{review_id}, application {application_reference} ({technology_label}), "
                        "reviewed under {standard_document} and opened by {coordinator_id}: "
                        "{status}.")
CHAT_SCREENING_TEMPLATE = ("Assessment {assessment}: {requirements}. Rating stated: "
                           "{rating_stated}. Network triggers: {triggers}. The decisions open: "
                           "{decisions_open}.")
CHAT_FIGURES_TEMPLATE = ("Network work likelihood {network_work_likelihood}, {likelihood_band}; "
                         "expected working days {expected_working_days}.")
CHAT_EVIDENCE_TEMPLATE = ("Statement evidence: {sentence_count} sentences read "
                          "({sentence_types}); equipment ratings flagged: {flag_count}.")
CHAT_NO_STATEMENTS_TEXT = "no statement is attached"
CHAT_NO_TRIGGERS_TEXT = "none"
CHAT_AWAITING_DECISION_TEMPLATE = ("It waits for a Connections Engineer's decision; the review "
                                   "recommends {recommended_decision}.")
CHAT_DECIDED_TEMPLATE = ("{decided_by}, {decided_by_role}, decided {decision}. The reason given: "
                         "\"{decision_reason}\"")
GROUP_SIZE_MINIMUM = 5
GROUP_SIZE_LINE = ("Figures about equipment at a substation are shared only where at least five "
                   "premises make them up.")
TECHNOLOGY_PLURALS = {
    EV_CHARGE_POINT: "EV charge points",
    HEAT_PUMP: "Heat pumps",
    VEHICLE_TO_GRID: "Vehicle-to-grid charge points",
    SOLAR_PV: "Solar PV installations",
    BATTERY_STORAGE: "Batteries",
}
CHAT_SUBSTATION_FIGURE_TEMPLATE = ("{technology_plural} are connected at {premises_count} premises "
                                   "on substation {substation_reference}.")
CHAT_SUBSTATION_NOT_NAMED_MESSAGE = ("Name one substation, as SS-0412, and one technology: EV "
                                     "charge points, heat pumps, vehicle-to-grid charge points, "
                                     "solar PV installations or batteries.")
PREMISES_REFERENCE_COLUMN = "premises_reference"

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8081"))
API_TITLE = "PhaseWise AI API"
API_BASE_URL = os.getenv("API_BASE_URL", f"http://localhost:{API_PORT}")
API_REQUEST_TIMEOUT_SECONDS = int(os.getenv("API_REQUEST_TIMEOUT_SECONDS", "300"))

DASHBOARD_TITLE = "PhaseWise AI"
DASHBOARD_SUBTITLE = f"{ORGANISATION_NAME} | Home low-carbon connections"
DASHBOARD_PAGE_STYLE = ('<style>input::-ms-reveal {display: none;} '
                        '[data-testid="stTextInput"] [data-testid="InputInstructions"] '
                        '{display: none;}</style>')
DECISION_BADGE_COLOURS = {DECISION_APPROVE: "green", DECISION_RETURN: "orange",
                          DECISION_REFER: "violet"}
STATUS_BADGE_COLOURS = {REVIEW_STATUS_AWAITING: "blue", REVIEW_STATUS_APPROVED: "green",
                        REVIEW_STATUS_RETURNED: "orange", REVIEW_STATUS_REFERRED: "violet"}
LIKELIHOOD_BAND_BADGE_COLOURS = {HIGH_LIKELIHOOD_BAND: "red", MEDIUM_LIKELIHOOD_BAND: "yellow",
                                 LOW_LIKELIHOOD_BAND: "green"}
CHAT_TYPE_BADGE_COLOURS = {CHAT_TYPE_CONNECTION_STANDARD: "blue",
                           CHAT_TYPE_CONNECTION_RULES: "violet",
                           CHAT_TYPE_STATEMENT_GUIDANCE: "green",
                           CHAT_TYPE_APPLICATION_STATUS: "orange",
                           CHAT_TYPE_OUTCOME_PREDICTION: "yellow",
                           CHAT_TYPE_SUBSTATION_FIGURE: "primary",
                           CHAT_TYPE_OUT_OF_SCOPE: "gray"}
NEUTRAL_BADGE_COLOUR = "gray"
HEATING_FAILED_BADGE_COLOUR = "red"
SENTENCE_TYPE_BADGE_COLOUR = "blue"
REQUIREMENT_RESULT_BADGE_COLOURS = {REQUIREMENT_MET: "green", REQUIREMENT_NOT_MET: "red",
                                    REQUIREMENT_NOT_DOCUMENTED: "orange"}
DELAY_COLOUR = "red"
LATENCY_SECONDS_DECIMALS = 1
