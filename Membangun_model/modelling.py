import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_basic():
    # 1. Muat Data (Pastikan file csv ada di folder dataset_preprocessing)
    # Path disesuaikan karena file ini berada di dalam folder Membangun_model
    try:
        df = pd.read_csv('dataset_preprocessing/cleaned_diabetes.csv')
    except FileNotFoundError:
        print("Error: File cleaned_diabetes.csv tidak ditemukan di folder dataset_preprocessing/")
        return

    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Setup MLflow Basic (Simpan secara lokal untuk membedakan dengan Advance)
    mlflow.set_experiment("Diabetes_Basic_Experiment")
    
    # KRITERIA BASIC: Menggunakan autolog
    mlflow.sklearn.autolog() 

    with mlflow.start_run(run_name="RandomForest_Standard"):
        # Menggunakan model standard tanpa tuning
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi sederhana
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"Model Basic Selesai. Akurasi: {acc}")

if __name__ == "__main__":
    train_basic()