import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import zipfile 

# Aktifkan auto-logging
mlflow.set_experiment("msml-basic")
mlflow.set_tracking_uri("file:./mlruns")
mlflow.sklearn.autolog()
# Ekstraksi dataset
zip_path = "../preprocessing/dataset_preprocessing.zip"  
    
with  zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('')

# Load data
df = pd.read_csv('dataset_preprocessing.csv')

# Pisahkan fitur dan target
X = df.drop(columns=['survived'])
y = df['survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow Tracking
with mlflow.start_run():
    model = RandomForestClassifier( random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc}")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model", input_example=X_test.iloc[:5])