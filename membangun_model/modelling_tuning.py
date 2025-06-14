import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import log_loss, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
from dagshub import DAGsHubLogger
import numpy as np
import os

load_dotenv()

#Setup MLflow DagsHub
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("msml-advance") 

# Setting kredensial login untuk DagsHub
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow.set_experiment("msml-advance")

df = pd.read_csv('dataset_preprocessing.csv')

#Preprocessing dasar
X = df.drop(columns=["survived"])
y = df["survived"]

# Pastikan semua fitur numerik
X = pd.get_dummies(X, drop_first=True)
X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Grid Search
params = {
     "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=params,  
    cv=3,
    scoring='accuracy',
    n_jobs=-1,  
    random_state=42
)

with mlflow.start_run():
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    #Manual logging (tanpa autolog)
    mlflow.log_params(grid.best_params_)
    y_proba = best_model.predict_proba(X_test)[:, 1]


    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    logloss = log_loss(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)  
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("log_loss", logloss)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("balanced_accuracy", balanced_acc)       
    mlflow.sklearn.log_model(best_model, "model", input_example= X_test.iloc[:5])

    #Logging ke DagsHub
    logger = DAGsHubLogger()
    logger.log_hyperparams(grid.best_params_)
    logger.log_metrics({
        "accuracy": acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    })

    print("Model terbaik:", grid.best_params_)
    print("Accuracy:", acc)
    