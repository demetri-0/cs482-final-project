import pandas as pd
from dython.nominal import associations

# prints number of features and samples for the data
def features_and_samples(data: pd.DataFrame):
    print(f"Number of Features: {data.shape[1] - 1}")
    print(f"Number of Samples: {data.shape[0]}")

# prints the number of UNKNOWN values in a feature alongside the percentage relative to total rows
def unknown_values(data: pd.DataFrame):
    unknown_counts = []

    for col in data.columns:
        unknown_counts.append((data[col] == "UNKNOWN").sum())

    unknown_percents_str = [f"{round(((unknown_count / len(data)) * 100), 2)}%" for unknown_count in unknown_counts]

    display_df = pd.DataFrame({
        "Feature Name": data.columns,
        "UNKNOWN Count": unknown_counts,
        "Percent UNKNOWN": unknown_percents_str
    })

    display_df = display_df.sort_values(by="UNKNOWN Count", ascending=False)

    print(display_df)

# prints all categories of categorical features, alonside their counts and percent relative to total rows
def category_counts(data: pd.DataFrame):
    categorical_features = data.select_dtypes(include=["object"]).columns
    total = len(data)

    for col in categorical_features:
        print(f"\n=== {col} ===")
        value_counts = data[col].value_counts(dropna=False)

        for category, count in value_counts.items():
            percent = (count / total) * 100
            print(f"{category}: {count} ({percent:.2f}%)")

# prints the an association matrix for categorical columns using Cramer's V
def cramers_v_analysis(data: pd.DataFrame):
    categorical_cols = data.select_dtypes(include=['object']).columns

    assoc_matrix = associations(
        data[categorical_cols],
        plot=False
    )['corr']

    print(assoc_matrix)
    