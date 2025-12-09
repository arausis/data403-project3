import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

holdout_map = {
    "TestSetImage01.png" : "Kelly",
    "TestSetImage02.png" : "Alex",
    "TestSetImage03.png" : "Alex",
    "TestSetImage04.png" : "Alex",
    "TestSetImage05.png" : "Alex",
    "TestSetImage06.png" : "Alex",
    "TestSetImage07.png" : "Kelly",
    "TestSetImage08.png" : "Kelly",
    "TestSetImage09.png" : "Kelly",
    "TestSetImage10.png" : "Kelly",
    "TestSetImage11.png" : "Kelly",
    "TestSetImage12.png" : "Alex",
    "TestSetImage13.png" : "Kelly",
    "TestSetImage14.png" : "Kelly",
    "TestSetImage15.png" : "Kelly",
    "TestSetImage16.png" : "Kelly",
    "TestSetImage17.png" : "Alex",
    "TestSetImage18.png" : "Alex",
    "TestSetImage19.png" : "Alex",
    "TestSetImage20.png" : "Alex"
}

path = os.path.dirname(os.path.abspath("composition_features.csv"))
comp = pd.read_csv(path + "/composition_features.csv")

content = pd.read_csv(path + "/content_features.csv")
content = content.rename(columns={'fname':'image'})

holdout_comp = pd.read_csv(f"{path}/holdout_composition_features.csv")
holdout_comp["image"] = holdout_comp["fname"]

holdout_content = pd.read_csv(f"{path}/holdout_content_features.csv")
holdout_content["image"] = holdout_content["fname"]

holdout_features = pd.merge(holdout_comp, holdout_content, on="image")
holdout_features["y"] = holdout_features["image"].map(lambda x: int(holdout_map[x] == "Kelly") )

holdout_features.drop(["fname_x", "fname_y"], inplace=True, axis=1)
holdout_features.drop(["who_took_x", "who_took_y"], inplace=True, axis=1)

features = pd.merge(comp, content, on = "image")
features["y"] = features["image"].str.contains("Kelly").astype(int)

print(features.columns)
print(holdout_features.columns)
exit()

features = pd.concat([holdout_features, features], ignore_index=True)
featuers = features[['image', 'tilted', 'clearFocalObject', 'vibrant', 'selfie','majoritySky', 'person', 'building', 'indoors', 'bodwinPeople', 'event','gameNight', 'sports', 'concert']]

X = features.drop(['image', 'y'], axis = 1)
y = features['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

l = [10000, 1000, 100, 10, 1, 0.1]
for i in l:
    model = LogisticRegression(penalty = "l1", solver = "saga", C = i, max_iter = 5000)
    model.fit(X_train_scaled, y_train)
    print("Train accuracy:", model.score(X_train_scaled, y_train))

model1 = LogisticRegression(penalty = "l1", solver = "saga", C = 1, max_iter = 5000)
model1.fit(X_train_scaled, y_train)

coefs = pd.DataFrame({
    "feature": features.drop(['image','y'], axis = 1).columns,
    "coef": model1.coef_[0]})
print(coefs)
