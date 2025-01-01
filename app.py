import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="KERJAAJA - Prediksi & Klasifikasi Level",
    page_icon="💼",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        margin-top: 1em;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('ds.csv')
    return df

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
        X[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    le = LabelEncoder()
    y = le.fit_transform(df['experience_level'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    return clf, scaler, le, le_dict

def perform_clustering(X_scaled, n_clusters=10):
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='random',
        max_iter=300,
        n_init=10,
        random_state=42
    )
    clusters = kmeans.fit_predict(X_scaled)
    return clusters

def main():
    st.title("💼 KERJAAJA")
    st.subheader("Sistem Prediksi Level Pengalaman dan Clustering")

    df = load_data()
    clf, scaler, le, le_dict = prepare_model(df)

    st.sidebar.title("Menu")
    page = st.sidebar.radio("Pilih Analisis:", ["Klasifikasi Level", "Clustering Data"])

    if page == "Klasifikasi Level":
        st.header("Klasifikasi Level Pengalaman")
        
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
                    index=0
                )
                
                salary_currency = st.selectbox(
                    "Mata Uang",
                    options=sorted(df['salary_currency'].unique()),
                    index=0
                )
                
                salary = st.number_input(
                    "Gaji",
                    min_value=int(df['salary'].min()),
                    max_value=int(df['salary'].max()),
                    value=int(df['salary'].mean())
                )
            
        with col2:
            employee_residence = st.selectbox(
                "Negara Karyawan",
                options=sorted(df['employee_residence'].unique()),
                index=0
            )
            
            remote_ratio = st.selectbox(
                "Remote Ratio",
                options=sorted(df['remote_ratio'].unique()),
                index=0
            )
            
            company_location = st.selectbox(
                "Lokasi Perusahaan",
                options=sorted(df['company_location'].unique()),
                index=0
            )
            
            company_size = st.selectbox(
                "Ukuran Perusahaan",
                options=sorted(df['company_size'].unique()),
                index=0
            )
            
            submitted = st.form_submit_button("Prediksi Level")
            
            if submitted:
                
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
                
                predicted_level = le.inverse_transform(prediction)[0]
                
                st.success("Hasil Prediksi Level Pengalaman:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Level Pengalaman", predicted_level)
                with col2:
                    st.metric("Confidence Score", f"{confidence:.2f}%")
                
                prob_df = pd.DataFrame({
                    'Level': le.classes_,
                    'Probability': probabilities[0] * 100
                })
                
                fig = px.bar(
                    prob_df, 
                    x='Level', 
                    y='Probability',
                    title='Distribusi Probabilitas Level Pengalaman'
                )
                st.plotly_chart(fig)

    else:  
        st.header("Analisis Clustering")
        
        features = ['job_title', 'salary']
        X = df[features].copy()
        
        le_job = LabelEncoder()
        X['job_title'] = le_job.fit_transform(X['job_title'])
        
        X_scaled = StandardScaler().fit_transform(X)
        
        n_clusters = st.slider("Jumlah Cluster", 2, 10, 5)
        
        clusters = perform_clustering(X_scaled, n_clusters)
        df_cluster = df.copy()
        df_cluster['Cluster'] = clusters

        st.subheader("Hasil Clustering")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        pca_df = pd.DataFrame(
            data=X_pca,
            columns=['PC1', 'PC2']
        )
        pca_df['Cluster'] = clusters
        
        fig3 = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='Cluster',
            title='Visualisasi Cluster berdasarkan Job Title dan Gaji'
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Statistik Ringkasan")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Statistik Gaji per Cluster")
            st.dataframe(df_cluster.groupby('Cluster')['salary'].describe())

        with col2:
            st.write("Rata-rata Gaji per Job Title")
            st.dataframe(df_cluster.groupby('job_title')['salary'].mean().sort_values(ascending=False).head())

if __name__ == "__main__":
    main() 