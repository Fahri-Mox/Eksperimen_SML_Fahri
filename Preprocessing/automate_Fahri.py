import pandas as pd
import os
import mlflow
from sklearn.preprocessing import StandardScaler

# --- Konfigurasi MLflow ---
# Menentukan nama eksperimen agar tercatat di MLflow UI
mlflow.set_experiment("Diabetes_Preprocessing_Fahri")

def run_preprocessing(input_filename, output_filename):
    """
    Fungsi otomatisasi preprocessing dengan tracking MLflow.
    Mencakup: Loading, Cleaning (Imputasi Median), Scaling, dan Logging.
    """
    
    # 1. Tentukan Path
    input_path = os.path.join('../dataset_raw', input_filename)
    output_dir = 'dataset_preprocessing'
    output_path = os.path.join(output_dir, output_filename)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Memulai tracking eksperimen dengan MLflow
    with mlflow.start_run(run_name="Preprocessing_Run"):
        print(f"Memulai proses otomatisasi untuk: {input_path}")
        
        # Log Parameter input ke MLflow
        mlflow.log_param("input_file", input_filename)
        mlflow.log_param("output_file", output_filename)

        try:
            # 2. Memuat Dataset
            df = pd.read_csv(input_path)
            mlflow.log_metric("raw_row_count", len(df))

            # 3. Handling Missing Values (Nilai 0 yang tidak logis)
            cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            for col in cols_to_fix:
                median_val = df[col].median()
                df[col] = df[col].replace(0, median_val)
                # Mencatat nilai median yang digunakan untuk setiap kolom
                mlflow.log_param(f"median_{col}", median_val)
            
            # 4. Menghapus Duplikat
            df = df.drop_duplicates()

            # 5. Fitur Scaling (Standarisasi)
            X = df.drop(columns=['Outcome'])
            y = df['Outcome']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Gabungkan kembali fitur yang sudah di-scale dengan target
            df_final = pd.DataFrame(X_scaled, columns=X.columns)
            df_final['Outcome'] = y.values

            # 6. Menyimpan Hasil Preprocessing secara Lokal
            df_final.to_csv(output_path, index=False)
            
            # 7. Mencatat Artifact (File Hasil) ke MLflow
            mlflow.log_artifact(output_path)
            mlflow.log_metric("final_row_count", len(df_final))
            
            print("-" * 30)
            print(f"SUKSES: Data berhasil diproses!")
            print(f"Lokasi file: {output_path}")
            print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
            print("-" * 30)

        except Exception as e:
            print(f"TERJADI KESALAHAN: {e}")
            mlflow.log_param("error", str(e))

if __name__ == "__main__":
    # Pastikan file diabetes.csv ada di folder ../dataset_raw/
    run_preprocessing('diabetes.csv', 'cleaned_diabetes.csv')