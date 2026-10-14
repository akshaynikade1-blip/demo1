import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

EXPERIMENT = "Food Delivery Delay Prediction"
MODEL_NAME = "DeliveryDelayClassifier"

# 1. Find the run with the highest f1 score in our experiment
experiment = client.get_experiment_by_name(EXPERIMENT)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1 DESC"],
    max_results=1,
)
best_run = runs[0]
print("Best run :", best_run.info.run_name)
print("f1       :", round(best_run.data.metrics["f1"], 3))

# 2. Get the saved model address (train.py stored it as a tag)
model_uri = best_run.data.tags["model_uri"]

# 3. Register it (first time = Version 1, next time = Version 2, ...)
registered = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
print("Registered model :", registered.name)
print("Model version    :", registered.version)

# 4. Give this version the alias "champion"
client.set_registered_model_alias(
    name=MODEL_NAME, alias="champion", version=registered.version
)
print("Alias 'champion' now points to version", registered.version)
