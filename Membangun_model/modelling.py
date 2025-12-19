import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_basic():
    # 1. Konfigurasi MLflow ke Localhost (Kriteria Skilled)
    # Pastikan Anda sudah menjalankan 'mlflow ui' di terminal
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Diabetes_Basic_Experiment")
    
    # 2. Muat Data
    try:
        df = pd.read_csv('dataset_preprocessing/cleaned_diabetes.csv')
    except FileNotFoundError:
        print("Error: File cleaned_diabetes.csv tidak ditemukan!")
        return

    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Logging Otomatis
    mlflow.sklearn.autolog() 

    with mlflow.start_run(run_name="RandomForest_Baseline_Local"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"Model Baseline Selesai. Akurasi: {acc}")

if __name__ == "__main__":
    train_basic()