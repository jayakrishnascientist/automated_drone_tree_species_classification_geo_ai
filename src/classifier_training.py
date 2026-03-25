
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_classifier(feature_path, label_csv):

    X = np.load(feature_path)
    df = pd.read_csv(label_csv)

    y = df["label"].map({
        "acacia":0,
        "non_acacia":1
    }).values

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train,y_train)

    preds = clf.predict(X_test)
    print(classification_report(y_test,preds))

    return clf
