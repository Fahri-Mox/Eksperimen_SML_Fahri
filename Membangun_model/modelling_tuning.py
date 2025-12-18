import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Hubungkan ke DagsHub (Ganti dengan username & nama repo Anda)
dagshub.init(repo_owner='Fahri-Mox', repo_name='Eksperimen_SML_Fahri', mlflow=True)

def train_advance():
    # 2. Muat Data Hasil Preprocessing
    df = pd.read_csv('dataset_preprocessing/cleaned_diabetes.csv')
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Mulai MLflow Experiment
    mlflow.set_experiment("Diabetes_Tuning_Advance")

    with mlflow.start_run(run_name="Advance_Hyperparameter_Tuning"):
        # 4. Hyperparameter Tuning (Kriteria Skilled/Advance)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        # 5. Manual Logging (Kriteria Advance)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)

        # 6. Menambahkan Artefak Tambahan (Minimal 2 untuk Advance)
        # Artefak 1: Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png") # Logging artefak pertama

        # Artefak 2: Simpan info dataset sederhana
        with open("dataset_info.txt", "w") as f:
            f.write(f"Jumlah baris data: {len(df)}")
        mlflow.log_artifact("dataset_info.txt") # Logging artefak kedua

        # Simpan Model
        mlflow.sklearn.log_model(best_model, "diabetes_model")
        
        print(f"Berhasil! Akurasi: {acc}. Cek Dashboard DagsHub Anda!")

if __name__ == "__main__":
    train_advance()