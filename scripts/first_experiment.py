
import mlflow

# Set the MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set the experiment name
mlflow.set_experiment("First MLflow Demo")

# Start one MLflow run
with mlflow.start_run(run_name="Demo Run 1"):

    # Log a parameter (setting we choose)
    mlflow.log_param("max_depth", 3)

    # Log a metric (result we measure)
    mlflow.log_metric("accuracy", 0.85)

    # Add a tag (description of the run)
    mlflow.set_tag(
        "description",
        "My first MLflow experiment"
    )

print("Experiment logged successfully! . Open http://127.0.0.1:5000")
