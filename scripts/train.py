import argparse
import subprocess

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 0. Read settings from the command line (so we can change them without editing code)
parser = argparse.ArgumentParser()
parser.add_argument("--max-depth", type=int, default=3)
parser.add_argument("--min-samples-leaf", type=int, default=1)
args = parser.parse_args()

# 1. Connect to MLflow and choose the experiment name
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Food Delivery Delay Prediction")

# 2. Load data
df = pd.read_csv("data/deliveries.csv")
features = ["distance_km", "raining", "heavy_traffic"]
X = df[features]
y = df["late_delivery"]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Find the current Git commit (optional; saved as a tag if Git is available)
try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()
except (OSError, subprocess.CalledProcessError):
    git_commit = "not_available"

# 5. Start an MLflow run - everything inside this block belongs to ONE run
with mlflow.start_run(run_name=f"tree_depth_{args.max_depth}"):

    # Log settings (parameters)
    mlflow.log_param("algorithm", "DecisionTree")
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
    mlflow.log_param("test_size", 0.2)

    # Log extra information (tags)
    mlflow.set_tag("git_commit", git_commit)
    mlflow.set_tag("dataset", "synthetic_deliveries")

    # 6. Create and train the model
    model = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 7. Evaluate on training data AND on unseen test data
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    accuracy = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred, zero_division=0)
    recall = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)

    # 8. Log results (metrics)
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    # 9. Save the trained model as an artifact
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="delivery_model",
        input_example=X_train.head(3),
        serialization_format="cloudpickle",   # needed for Decision Trees on newer MLflow
    )

    # Remember the model address so the registry script can find it later
    mlflow.set_tag("model_uri", model_info.model_uri)

    print("-" * 50)
    print("max_depth      :", args.max_depth)
    print("train accuracy :", round(train_accuracy, 3))
    print("test accuracy  :", round(accuracy, 3))
    print("precision      :", round(precision, 3))
    print("recall         :", round(recall, 3))
    print("f1             :", round(f1, 3))
    print("Model URI      :", model_info.model_uri)
    print("-" * 50)