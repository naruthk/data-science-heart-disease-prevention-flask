#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:44:16 2020

@author: nk
"""

import pandas as pd
import medml as ml
import pickle

column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                "restecg", "thalach", "exang", "oldpeak", "slope",
                "ca", "thal", "num"]

df = pd.read_csv("processed.cleveland.data", header=None, names=column_names)

df = df[df["thal"] != "?"].reset_index(drop=True)
df = df[df["ca"] != "?"].reset_index(drop=True)

df["labels"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

features = ["thal", "exang", "cp", "ca", "slope"]

# thal
thal = pd.get_dummies(df["thal"])
thal.columns = ["normal", "fixed defect", "reversable defect"]
df = pd.concat([df, thal], axis=1)

# cp
df["cp"] = df["cp"].map({1: "typical angina",
                         2: "atypical angina",
                         3: "non-anginal pain",
                         4: "asymptomatic"})
cp = pd.get_dummies(df["cp"])
df = pd.concat([df, cp], axis=1)

# slope
df["slope"] = df["slope"].map({1: "upsloping",
                               2: "flat",
                               3: "downsloping"})
slope = pd.get_dummies(df["slope"])
df = pd.concat([df, slope], axis=1)

data = df.loc[:, ["normal", "fixed defect", "reversable defect", "typical angina",
                  "atypical angina", "non-anginal pain", "asymptomatic",
                  "upsloping", "flat", "downsloping", "exang", "ca", "labels"]]

features = data.columns.tolist()
features.remove("labels")

params = {"dt": {"max_depth": 3},
          "lr": {"penalty": "l2"}}

heart = ml.ai(data=df,
              features=features,
              target="labels",
              test_size=0.2,
              model="lr",
              params=params)

print(heart.clf.coef_)

# Save to file
pickle.dump(heart.clf, open("Logregheart.pkl", "wb"))

# Load from file
myheart = pickle.load(open("Logregheart.pkl", "rb"))

print(myheart.coef_)