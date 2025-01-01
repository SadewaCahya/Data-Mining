from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Function to load your data (make sure you adjust this part based on your dataset)
def load_data():
    # Load the dataset here, e.g., df = pd.read_csv("your_data.csv")
    df = pd.read_csv("ds.csv")
    return df

# Function to prepare the model, scaler, and label encoders
def prepare_model(df):
    # Label encoding for categorical columns
    label_encoders = {}
    for column in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Split the dataset into features (X) and target (y)
    X = df.drop(columns=["experience_level"])  # Adjust based on your target column
    y = df["experience_level"]

    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")

    return clf, scaler, label_encoders

# Function for clustering (optional if you want to include this)
def perform_clustering(X_scaled, n_clusters):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    return clusters

# Streamlit main application
def main():
    # Header
    st.title("💼 KERJAA KERJAA")
    st.subheader("Sistem Prediksi Level Pengalaman dan Clustering")

    # Load and prepare data
    df = load_data()
    
    # Prepare the model, scaler, and label encoders
    clf, scaler, label_encoders = prepare_model(df)

    # Sidebar for navigation
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Pilih Analisis:", ["Klasifikasi Level", "Clustering Data"])

    if page == "Klasifikasi Level":
        st.header("Klasifikasi Level Pengalaman")
        
        # Input form
        with st.form("classification_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                work_year = st.selectbox(
                    "Tahun Kerja",
                    options=sorted(df['work_year'].unique()),
                    index=list(sorted(df['work_year'].unique())).index(2023)
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
            
            submitted = st.form_submit_button("Prediksi Level")
            
            if submitted:
                # Apply label encoding for user inputs
                encoded_inputs = []
                for feature, le in label_encoders.items():
                    if feature == "work_year":
                        encoded_inputs.append(work_year)
                    elif feature == "employment_type":
                        encoded_inputs.append(le.transform([employment_type])[0])
                    elif feature == "job_title":
                        encoded_inputs.append(le.transform([job_title])[0])
                    elif feature == "salary_currency":
                        encoded_inputs.append(le.transform([salary_currency])[0])
                    elif feature == "employee_residence":
                        encoded_inputs.append(le.transform([employee_residence])[0])
                    elif feature == "company_location":
                        encoded_inputs.append(le.transform([company_location])[0])
                    elif feature == "company_size":
                        encoded_inputs.append(le.transform([company_size])[0])
                    else:
                        encoded_inputs.append(salary)  # For numeric features, just append the salary
                
                # Convert input to numpy array and scale it
                input_data = np.array([encoded_inputs])
                input_scaled = scaler.transform(input_data)
                
                # Make prediction
                prediction = clf.predict(input_scaled)
                probabilities = clf.predict_proba(input_scaled)
                confidence = np.max(probabilities) * 100
                
                # Get predicted level
                predicted_level = label_encoders["experience_level"].inverse_transform(prediction)[0]
                
                # Show results
                st.success("Hasil Prediksi Level Pengalaman:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Level Pengalaman", predicted_level)
                with col2:
                    st.metric("Confidence Score", f"{confidence:.2f}%")
                
                # Show probability distribution
                prob_df = pd.DataFrame({
                    'Level': label_encoders["experience_level"].classes_,
                    'Probability': probabilities[0] * 100
                })
                
                fig = px.bar(
                    prob_df, 
                    x='Level', 
                    y='Probability',
                    title='Distribusi Probabilitas Level Pengalaman'
                )
                st.plotly_chart(fig)

    else:  # Clustering page
        st.header("Analisis Clustering")
        
        # Prepare data for clustering
        features = ['job_title', 'salary']
        X = df[features].copy()
        
        # Perform clustering
        clusters = perform_clustering(X_scaled, n_clusters)
        df_cluster = df.copy()
        df_cluster['Cluster'] = clusters

        # Clustering visualization
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

        # Summary statistics
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
