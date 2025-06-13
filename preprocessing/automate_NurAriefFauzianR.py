import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import zipfile
import os
import sys

def preprocess_dataset(zip_path, output_path, csv_inside_zip='dataset_med.csv', target_column='survived'):
    # Ekstrak file ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("extracted_dataset")

    # Baca CSV dari dalam folder hasil ekstraksi
    df = pd.read_csv(f"extracted_dataset/{csv_inside_zip}")

    # Tangani missing value
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Hapus kolom yang tidak berguna
    drop_cols = ['id', 'diagnosis_date', 'end_treatment_date']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Hapus outlier dengan metode IQR
    def remove_outliers_iqr(data, columns):
        for col in columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            data = data[(data[col] >= lower) & (data[col] <= upper)]
        return data

    numerical_cols = ['age', 'bmi', 'cholesterol_level']
    df = remove_outliers_iqr(df, numerical_cols)

    # Pisahkan fitur dan target
    X = df.drop(columns=target_column)
    y = df[target_column]

    # One-hot encoding kolom kategorikal
    X = pd.get_dummies(X,
                       columns=['gender','smoking_status', 'treatment_type', 'family_history', 'cancer_stage', 'country'],
                       drop_first=True)
     

    # Standardisasi fitur numerik
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    #Resampling
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    # Gabungkan kembali dan simpan ke output
    processed_df = pd.concat([X_resampled, y_resampled.reset_index(drop=True)], axis=1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print(f"Preprocessing selesai dan disimpan ke {output_path}")

# Untuk dipanggil dari GitHub Actions (via CLI)
if __name__ == "__main__":
    zip_input = sys.argv[1]              
    csv_inside_zip = sys.argv[2]         
    output_path = sys.argv[3]            
    preprocess_dataset(zip_input, output_path, csv_inside_zip)
