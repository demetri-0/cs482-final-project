import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data, build_preprocessor
from sklearn.pipeline import Pipeline

raw_data = pd.read_csv("traffic_accidents.csv")

prepped_data = preprocess_data(raw_data)

X = prepped_data.drop(columns=["injury_severity"], axis=1)
y = prepped_data["injury_severity"]
x_train, x_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25, 
                                                    shuffle=True, 
                                                    random_state=42)

preprocessor = build_preprocessor(x_train)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor)
        # model stuff will go here
    ]
)
