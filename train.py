import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Tracking DB
mlflow.set_experiment("titanic-experiment")
mlflow.set_registry_uri("gs://dvc-mlops-storage-blah/mlflow")  # GCS artifact store

# Start MLflow run
with mlflow.start_run():

    df = pd.read_csv("data/titanic.csv")

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    X = df[["Pclass", "Sex", "Age", "Fare"]]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter
    max_iter = 500
    mlflow.log_param("max_iter", max_iter)

    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print("Accuracy:", acc)

    # Log metric
    mlflow.log_metric("accuracy", acc)

    # Save model
    joblib.dump(model, "model.pkl")

    # Log model artifact
    mlflow.sklearn.log_model(model, "model")
