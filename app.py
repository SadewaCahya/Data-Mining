import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pycountry import countries

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
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    h1 {
        color: #1E88E5;
        font-family: 'Helvetica Neue', sans-serif;
        padding-bottom: 20px;
    }
    h2 {
        color: #333;
        font-family: 'Helvetica Neue', sans-serif;
        padding: 20px 0 10px 0;
    }
    h3 {
        color: #666;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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

def perform_clustering(X_scaled, n_clusters=5):
    
    from sklearn.metrics import silhouette_score
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',  
        max_iter=300,
        n_init=10,
        random_state=42
    )
    clusters = kmeans.fit_predict(X_scaled)
    
    silhouette_avg = silhouette_score(X_scaled, clusters)
    
    return clusters, kmeans, silhouette_avg

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
                    index=list(sorted(df['work_year'].unique())).index(2023)
                )
                
                
                employment_map = {
                    'FT': 'Full Time',
                    'PT': 'Part Time',
                    'CT': 'Contract',
                    'FL': 'Freelance'
                }
                employment_options = [employment_map[et] for et in sorted(df['employment_type'].unique())]
                employment_type = st.selectbox(
                    "Tipe Pekerjaan",
                    options=employment_options,
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
                
                def get_country_name(code):
                    try:
                        return countries.get(alpha_2=code).name
                    except:
                        return code
                
                employee_residence_options = [get_country_name(code) for code in sorted(df['employee_residence'].unique())]
                employee_residence = st.selectbox(
                    "Negara Karyawan",
                    options=employee_residence_options,
                    index=employee_residence_options.index('United States') if 'United States' in employee_residence_options else 0
                )
                
                remote_map = {
                    0: 'No Remote (On-site)',
                    50: 'Partially Remote',
                    100: 'Fully Remote'
                }
                remote_options = [remote_map[r] for r in sorted(df['remote_ratio'].unique())]
                remote_ratio = st.selectbox(
                    "Tipe Kerja Remote",
                    options=remote_options,
                    index=0
                )
                
                company_location_options = [get_country_name(code) for code in sorted(df['company_location'].unique())]
                company_location = st.selectbox(
                    "Lokasi Perusahaan",
                    options=company_location_options,
                    index=company_location_options.index('United States') if 'United States' in company_location_options else 0
                )
                
                size_map = {
                    'S': 'Small',
                    'M': 'Medium',
                    'L': 'Large'
                }
                company_size_options = [size_map[s] for s in sorted(df['company_size'].unique())]
                company_size = st.selectbox(
                    "Ukuran Perusahaan",
                    options=company_size_options,
                    index=0
                )
            
            submitted = st.form_submit_button("Prediksi Level")
            
            if submitted:
                
                size_map_reverse = {
                    'Small': 'S',
                    'Medium': 'M',
                    'Large': 'L'
                }
                employment_map_reverse = {
                    'Full Time': 'FT',
                    'Part Time': 'PT',
                    'Contract': 'CT',
                    'Freelance': 'FL'
                }
                remote_map_reverse = {
                    'No Remote (On-site)': 0,
                    'Partially Remote': 50,
                    'Fully Remote': 100
                }

                company_size_orig = size_map_reverse[company_size]
                employment_type_orig = employment_map_reverse[employment_type]
                remote_ratio_orig = remote_map_reverse[remote_ratio]

                input_data = np.array([[
                    work_year, 
                    salary,
                    remote_ratio_orig,  
                    le_dict['company_size'].transform([company_size_orig])[0],  
                    le_dict['employment_type'].transform([employment_type_orig])[0]  
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
        st.header("📊 Analisis Clustering")
        
        features = ['job_title', 'salary']
        X = df[features].copy()
        
        le_job = LabelEncoder()
        X['job_title'] = le_job.fit_transform(X['job_title'])
        
        X_scaled = StandardScaler().fit_transform(X)
        
        col1, col2 = st.columns([2,1])
        with col1:
            n_clusters = st.slider(
                "Jumlah Cluster",
                min_value=2,
                max_value=10,
                value=5,
                help="Geser untuk mengubah jumlah kelompok"
            )
        
        clusters, kmeans, silhouette_avg = perform_clustering(X_scaled, n_clusters)
        df_cluster = df.copy()
        df_cluster['Cluster'] = clusters

        with col2:
            st.metric(
                "Silhouette Score",
                f"{silhouette_avg:.3f}",
                help="Skor 1 menunjukkan clustering terbaik, -1 terburuk"
            )

        st.subheader("Visualisasi Hasil Clustering")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Visualisasi Cluster', 'Distribusi Gaji per Cluster'),
            specs=[[{'type': 'scatter'}, {'type': 'box'}]]
        )
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        colors = px.colors.qualitative.Set3[:n_clusters]
        
        for i in range(n_clusters):
            mask = clusters == i
            fig.add_trace(
                go.Scatter(
                    x=X_pca[mask, 0],
                    y=X_pca[mask, 1],
                    mode='markers',
                    name=f'Cluster {i}',
                    marker=dict(size=8, color=colors[i]),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        fig.add_trace(
            go.Box(
                y=df_cluster['salary'],
                x=df_cluster['Cluster'].astype(str),
                name='Gaji',
                marker_color='rgb(107,174,214)'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="Analisis Cluster dan Distribusi Gaji",
            showlegend=True,
            template='plotly_white',
            boxmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 