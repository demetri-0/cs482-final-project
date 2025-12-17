import pandas as pd
import preprocess_helper as ph
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# uncomment as needed - used to show metrics
# import evaluation as eval

# apply preprocessing functions to the raw data, returning prepped data
def preprocess_data(raw_data: pd.DataFrame) -> pd.DataFrame:

    raw_data_copy = raw_data.copy()

    ph.drop_leaky_columns(raw_data_copy)
    ph.drop_identifier_columns(raw_data_copy)
    # uncomment as needed - used to show metrics
    # eval.unknown_values(raw_data_copy.drop(columns=["most_severe_injury"], axis=1))
    # eval.cramers_v_analysis(raw_data_copy.drop(columns=["most_severe_injury"]))
    ph.association_analysis(raw_data_copy)
    ph.combine_categories(raw_data_copy)
    ph.build_target(raw_data_copy)

    prepped_data = raw_data_copy

    return prepped_data

# build and return a transformer to encode categorical features and scale numeric features
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:

    categorical_features = X.select_dtypes(include=["object"]).columns
    numeric_features = X.select_dtypes(include=["number"]).columns 

    categorical_transformer = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numeric", numeric_transformer, numeric_features)
        ]
    )

    return preprocessor
