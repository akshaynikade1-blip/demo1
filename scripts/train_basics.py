import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load the data
df = pd.read_csv("data/deliveries.csv")

# 2. Separate features (X = inputs) and label (y = answer)
features = ["distance_km", "raining", "heavy_traffic"]
X = df[features]
y = df["late_delivery"]

# 3. Split: 80% to learn from, 20% kept hidden for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training orders:", len(X_train))
print("Testing orders :", len(X_test))

# 4. Create the model and train it
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)                  # fit = learn from data

# 5. Predict on the hidden test orders and measure accuracy
predictions = model.predict(X_test)          # predict = make predictions
accuracy = accuracy_score(y_test, predictions)
print("Model accuracy:", round(accuracy, 3))

# 6. Predict ONE brand new order: 10 km, raining, heavy traffic
new_order = pd.DataFrame([[10, 1, 1]], columns=features)
result = model.predict(new_order)[0]
print("New order prediction:", "LATE" if result == 1 else "ON TIME")
