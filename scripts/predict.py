import mlflow
import mlflow.sklearn
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model = mlflow.sklearn.load_model("models:/DeliveryDelayClassifier@champion")

orders = pd.DataFrame([
    {"distance_km": 10, "raining": 1, "heavy_traffic": 1},   # long, rain, traffic
    {"distance_km": 2,  "raining": 0, "heavy_traffic": 0},   # short, clear roads
])

predictions = model.predict(orders)

for i, p in enumerate(predictions):
    print(f"Order {i + 1}:", orders.iloc[i].to_dict(), "->", "LATE" if p == 1 else "ON TIME")
