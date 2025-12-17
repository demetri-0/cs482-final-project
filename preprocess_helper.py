import pandas as pd
from dython.nominal import associations

# drops features deemed to be leaky from the data
def drop_leaky_columns(raw_data: pd.DataFrame):

    leaky_column_names = [
        "injuries_total",
        "injuries_fatal",
        "injuries_incapacitating",
        "injuries_non_incapacitating",
        "injuries_reported_not_evident",
        "injuries_no_indication",
        "crash_type",
        "damage"
    ]

    raw_data.drop(columns=leaky_column_names, axis=1, inplace=True)

# drops features deemed to be identifiers from the data
def drop_identifier_columns(raw_data: pd.DataFrame):

    identifier_columns = [
        "crash_date"
    ]

    raw_data.drop(columns=identifier_columns, axis=1, inplace=True)

# performs an association analysis on the data, dropping features with too strong of an association
def association_analysis(raw_data: pd.DataFrame):
    raw_data = raw_data.drop(columns=["most_severe_injury"], axis=1) # drop target column
    cat_cols = raw_data.select_dtypes(include=['object']).columns # obtain categorical columns

    # create association matrix
    assoc_matrix = associations(
        raw_data[cat_cols],
        plot=False
    )['corr']

    # iterate through association matrix, adding the first feature of a pair with strong association to a set that will be dropped from the data
    columns = assoc_matrix.columns
    cols_to_drop = set()
    for i in range(0, len(columns)):
        for j in range(i + 1, len(columns)):
            value = assoc_matrix.iloc[i, j]
            if value > 0.75:
                cols_to_drop.add(columns[i])

    # drop features with a strong association
    for col in cols_to_drop:
        raw_data = raw_data.drop(columns=[col], axis=1)

# based on a designated mapping for certain categorical features, combine categories into existing or new categories
def combine_categories(raw_data: pd.DataFrame):

    traffic_device_mapping = {
        "RAILROAD CROSSING GATE": "RAILROAD CONTROL",
        "OTHER RAILROAD CROSSING": "RAILROAD CONTROL",
        "RR CROSSING SIGN": "RAILROAD CONTROL",

        "OTHER REG. SIGN": "OTHER SIGN",
        "OTHER WARNING SIGN": "OTHER SIGN",
        "PEDESTRIAN CROSSING SIGN": "OTHER SIGN",
        "NO PASSING": "OTHER SIGN",
        "BICYCLE CROSSING SIGN": "OTHER SIGN",

        "DELINEATORS": "OTHER",
        "LANE USE MARKING": "OTHER",
        "POLICE/FLAGMAN": "OTHER",
        "SCHOOL ZONE": "OTHER",

        "FLASHING CONTROL SIGNAL": "TRAFFIC SIGNAL"
    }
    raw_data["traffic_control_device"] = raw_data["traffic_control_device"].replace(traffic_device_mapping)

    weather_mapping = {
        "BLOWING SNOW": "SNOW",

        "SLEET/HAIL": "FREEZING RAIN/DRIZZLE",

        "BLOWING SAND, SOIL, DIRT": "FOG/SMOKE/HAZE"
    }
    raw_data["weather_condition"] = raw_data["weather_condition"].replace(weather_mapping)

    first_crash_mapping = {
        "REAR TO FRONT": "BACKUP COLLISION",
        "REAR TO SIDE": "BACKUP COLLISION",
        "REAR TO REAR": "BACKUP COLLISION",

        "TRAIN": "OTHER OBJECT"
    }
    raw_data["first_crash_type"] = raw_data["first_crash_type"].replace(first_crash_mapping)

    trafficway_mapping = {
        "T-INTERSECTION": "NON-FOUR WAY INTERSECTION",
        "Y-INTERSECTION": "NON-FOUR WAY INTERSECTION",
        "L-INTERSECTION": "NON-FOUR WAY INTERSECTION",

        "NOT REPORTED": "UNKNOWN"
    }
    raw_data["trafficway_type"] = raw_data["trafficway_type"].replace(trafficway_mapping)

    prim_cause_mapping = {
        "FAILING TO REDUCE SPEED TO AVOID CRASH": "SPEEDING RELATED",
        "EXCEEDING SAFE SPEED FOR CONDITIONS": "SPEEDING RELATED",
        "EXCEEDING AUTHORIZED SPEED LIMIT": "SPEEDING RELATED",

        "DISTRACTION - FROM OUTSIDE VEHICLE": "DISTRACTION",
        "DISTRACTION - FROM INSIDE VEHICLE": "DISTRACTION",
        "CELL PHONE USE OTHER THAN TEXTING": "DISTRACTION",
        "DISTRACTION - OTHER ELECTRONIC DEVICE (NAVIGATION DEVICE, DVD PLAYER, ETC.)": "DISTRACTION",
        "TEXTING": "DISTRACTION",

        "UNDER THE INFLUENCE OF ALCOHOL/DRUGS (USE WHEN ARREST IS EFFECTED)": "SUBSTANCE IMPAIREMENT",
        "HAD BEEN DRINKING (USE WHEN ARREST IS NOT MADE)": "SUBSTANCE IMPAIREMENT",

        "VISION OBSCURED (SIGNS, TREE LIMBS, BUILDINGS, ETC.)": "NONSUBSTANCE/DRIVER IMPAIREMENT",
        "EQUIPMENT - VEHICLE CONDITION": "NONSUBSTANCE/DRIVER IMPAIREMENT",
        "PHYSICAL CONDITION OF DRIVER": "NONSUBSTANCE/DRIVER IMPAIREMENT",
        
        "ANIMAL": "OTHER EXTERNAL FACTORS",
        "OBSTRUCTED CROSSWALKS": "OTHER EXTERNAL FACTORS",
        "BICYCLE ADVANCING LEGALLY ON RED LIGHT": "OTHER EXTERNAL FACTORS",
        "MOTORCYCLE ADVANCING LEGALLY ON RED LIGHT": "OTHER EXTERNAL FACTORS",
        "ROAD CONSTRUCTION/MAINTENANCE": "OTHER EXTERNAL FACTORS",
        "EVASIVE ACTION DUE TO ANIMAL, OBJECT, NONMOTORIST": "OTHER EXTERNAL FACTORS",
        "ROAD ENGINEERING/SURFACE/MARKING DEFECTS": "OTHER EXTERNAL FACTORS",
        "RELATED TO BUS STOP": "OTHER EXTERNAL FACTORS"
    }
    raw_data["prim_contributory_cause"] = raw_data["prim_contributory_cause"].replace(prim_cause_mapping)

    injury_mapping = {
        "REPORTED, NOT EVIDENT": "NONINCAPACITATING INJURY",

        "INCAPACITATING INJURY": "INCAPACITATING/FATAL INJURY",
        "FATAL": "INCAPACITATING/FATAL INJURY"
    }
    raw_data["most_severe_injury"] = raw_data["most_severe_injury"].replace(injury_mapping)

# after the target column has been consolidated, values are mapped to numbers and the target is renamed
def build_target(raw_data: pd.DataFrame):

    target_mapping = {
        "NO INDICATION OF INJURY": 0,
        "NONINCAPACITATING INJURY": 1,
        "INCAPACITATING/FATAL INJURY": 2
    }

    raw_data["most_severe_injury"] = raw_data["most_severe_injury"].replace(target_mapping)

    raw_data.rename(columns={"most_severe_injury": "injury_severity"}, inplace=True)