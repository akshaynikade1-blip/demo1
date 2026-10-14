import numpy as np   # declared as np
import pandas as pd  #declared as pd

rng = np.random.default_rng(42)
n = 1000     #total 1000 deliveries

# features
distance = rng.uniform(1, 15, n)    # create random distnces from 1 to 15 till 1000 deliveries
raining = rng.integers(0, 2, n)  # raining between 0 and 1 0 = not raining 1 = raining
heavy_traffic = rng.integers(0, 2, n)  # heavy traffic between 0 and 1 0 = no traffic and 1 = traffic

# invent rule longer  distnace and rain and traffic make a late delivery more likely
score = -3.0 + +0.20*distance + 0.90 * raining + 1.20 * heavy_traffic
probability = 1/(1+np.exp(-score))  # turns store into a probability ( 0 to 1)

# label
late_delivery = rng.binomial(1, probability)

df = pd.DataFrame({
    "order_id": range(1, n+1),
    "distance_km": distance.round(2),
    "raining": raining,
    "heavy_traffic": heavy_traffic,
    "late_delivery": late_delivery
})


df.to_csv("data/deliveries.csv", index=False)

print("Dataset created : ", df.shape)
print(df.head())
print()
print("late deliveries :", int(df["late_delivery"].sum()))
print("on time deliveries :", int((df["late_delivery"] == 0).sum()))
