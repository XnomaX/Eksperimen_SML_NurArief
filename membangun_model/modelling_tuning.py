import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
from dagshub import DAGsHubLogger
import numpy as np

#Setup MLflow DagsHub
mlflow.set_tracking_uri("https://dagshub.com/nurarief1123/msml-nurarief.mlflow")
mlflow.set_experiment("msml-advance")

df = pd.read_csv('dataset_preprocessing.csv')

# Ambil sampel acak 500 baris dari dataset untuk mempercepat proses training (komputasi terbatas)
df = df.sample(n=500, random_state=42)

#Preprocessing dasar
X = df.drop(columns=["survived"])
y = df["survived"]

# Pastikan semua fitur numerik
X = pd.get_dummies(X, drop_first=True)
X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Grid Search
params = {
     "n_estimators": np.linspace(10, 1000, 5, dtype=int),
    "max_depth": np.linspace(1, 50, 5, dtype=int)
}

grid = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=params,
    n_iter=6,  
    cv=3,
    scoring='accuracy',
    n_jobs=2,  
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
    