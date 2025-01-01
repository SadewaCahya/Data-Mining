import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('ds.csv')
    except FileNotFoundError:
        st.error("File 'ds.csv' tidak ditemukan.")
        return None  # Menghentikan proses jika file tidak ditemukan
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file CSV: {e}")
        return None  # Menghentikan proses jika ada kesalahan lainnya
    
    # Proses penggantian nilai
    df['experience_level'] = df['experience_level'].replace({
        'SE': 'Senior',
        'EN': 'Entry level',
        'EX': 'Executive level',
        'MI': 'Mid/Intermediate level',
    })
    df['employment_type'] = df['employment_type'].replace({
        'FL': 'Freelancer',
        'CT': 'Contractor',
        'FT' : 'Full-time',
        'PT' : 'Part-time'
    })
    df['company_size'] = df['company_size'].replace({
        'S': 'SMALL',
        'M': 'MEDIUM',
        'L' : 'LARGE',
    })
    df['remote_ratio'] = df['remote_ratio'].astype(str)
    df['remote_ratio'] = df['remote_ratio'].replace({
        '0': 'On-Site',
        '50': 'Half-Remote',
        '100' : 'Full-Remote',
    })
    return df

# Fungsi untuk menyiapkan model
@st.cache_data
def prepare_model(df):
    X = df[[
        'work_year', 'salary', 'remote_ratio', 
        'company_size', 'employment_type'
    ]]
    
    le_dict = {}
    categorical_cols = ['company_size', 'employment_type']
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Mengisi nilai kosong jika ada
        if df[col].isnull().any():
            df[col].fillna('Unknown', inplace=True)
        X[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    le = LabelEncoder()
    y = le.fit_transform(df['experience_level'])
    
    # Standarisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    return clf, scaler, le, le_dict

# Fungsi untuk mempersiapkan input dan prediksi
def make_prediction(work_year, salary, remote_ratio, company_size, employment_type, clf, scaler, le_dict):
    input_data = np.array([[
        work_year, 
        salary,
        remote_ratio,
        le_dict['company_size'].transform([company_size])[0],
        le_dict['employment_type'].transform([employment_type])[0]
    ]])
    
    input_scaled = scaler.transform(input_data)
    prediction = clf.predict(input_scaled)
    probabilities = clf.predict_proba(input_scaled)
    confidence = np.max(probabilities) * 100

    predicted_level = le_dict['experience_level'].inverse_transform(prediction)[0]
    
    return predicted_level, confidence

# Memuat data
df = load_data()
if df is not None:
    # Siapkan model
    clf, scaler, le, le_dict = prepare_model(df)

    # Form untuk input pengguna
    with st.form("classification_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            work_year = st.selectbox(
                "Tahun Kerja",
                options=sorted(df['work_year'].unique()),
                index=list(sorted(df['work_year'].unique())).index(2023) if 2023 in df['work_year'].unique() else 0
            )
            employment_type = st.selectbox(
                "Tipe Pekerjaan",
                options=sorted(df['employment_type'].unique()),
                index=0
            )
            job_title = st.selectbox(
                "Posisi",
                options=sorted(df['job_title'].unique()),
                index=list(sorted(df['job_title'].unique())).index('Machine Learning Engineer') if 'Machine Learning Engineer' in df['job_title'].unique() else 0
            )
            salary_currency = st.selectbox(
                "Mata Uang",
                options=sorted(df['salary_currency'].unique()),
                index=list(sorted(df['salary_currency'].unique())).index('USD') if 'USD' in df['salary_currency'].unique() else 0
            )
            salary = st.number_input(
                "Gaji",
                min_value=int(df['salary'].min()),
                max_value=int(df['salary'].max()),
                value=121523
            )
        
        with col2:
            employee_residence = st.selectbox(
                "Negara Karyawan",
                options=sorted(df['employee_residence'].unique()),
                index=list(sorted(df['employee_residence'].unique())).index('US') if 'US' in df['employee_residence'].unique() else 0
            )
            remote_ratio = st.selectbox(
                "Remote Ratio",
                options=sorted(df['remote_ratio'].unique()),
                index=0
            )
            company_location = st.selectbox(
                "Lokasi Perusahaan",
                options=sorted(df['company_location'].unique()),
                index=list(sorted(df['company_location'].unique())).index('US') if 'US' in df['company_location'].unique() else 0
            )
            company_size = st.selectbox(
                "Ukuran Perusahaan",
                options=sorted(df['company_size'].unique()),
                index=list(sorted(df['company_size'].unique())).index('M') if 'M' in df['company_size'].unique() else 0
            )
        
        # Ketika form disubmit
        if st.form_submit_button("Prediksi"):
            # Membuat prediksi
            predicted_level, confidence = make_prediction(
                work_year, salary, remote_ratio, company_size, employment_type,
                clf, scaler, le_dict
            )
            
            # Menampilkan hasil
            st.success("Hasil Prediksi Level Pengalaman:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Level Pengalaman", predicted_level)
            with col2:
                st.metric("Confidence Score", f"{confidence:.2f}%")
