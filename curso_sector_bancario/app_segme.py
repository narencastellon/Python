import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os  # Para chequeo de archivo

# Configuración de la página para estilo profesional y moderno (inspirado en Tableau: fondos limpios, tipografía sans-serif, sombras suaves, colores neutros con acentos azules)
st.set_page_config(page_title="Segmentación de Clientes Bancarios", layout="wide", initial_sidebar_state="expanded")

# CSS personalizado para estilo Tableau-like: fondo blanco/gris, elementos con bordes redondeados, sombras, colores azul/gris
st.markdown("""<style>
    .main { background-color: #f0f4f8; }  /* Fondo principal cambiado a gris claro suave para mejor visibilidad de textos */
    .stButton>button { background-color: #0056b3; color: white; border-radius: 5px; transition: background-color 0.3s; }
    .stButton>button:hover { background-color: #003f88; }
    .stSlider .stSliderLabel { color: #333333; font-family: Arial, sans-serif; }
    .stSelectbox { background-color: #6fab6c; border: 1px solid #ced4da; border-radius: 5px; }
    h1, h2, h3 { color: #1a1a1a; font-family: Arial, sans-serif; }
    .sidebar .sidebar-content { background-color: #e9ecef; box-shadow: 2px 0 5px rgba(0,0,0,0.1); }  /* Fondo sidebar gris medio para contraste */
    .block-container { padding: 20px; background-color: "#6fab6c"; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    body { color: #333333; }  /* Asegurar texto oscuro para legibilidad */
</style>
""", unsafe_allow_html=True)

# Cargar datos
@st.cache_data
def load_data(path= './bank_customers_segmentation.csv'):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            return pd.DataFrame()
    else:
        st.error(f"Archivo '{path}' no encontrado. Asegúrate de generarlo primero.")
        return pd.DataFrame()
    return df

bank_df = load_data()

# Preprocesamiento y entrenamiento del modelo K-Means (cacheado)
@st.cache_resource
def preprocess_and_cluster():
    # Codificar categóricas
    cat_cols = ['gender', 'region', 'preferred_channel']
    for col in cat_cols:
        bank_df[col] = LabelEncoder().fit_transform(bank_df[col])
    
    # Escalar features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(bank_df.drop(columns=['customer_id']))
    
    # K-Means con K óptimo (asumimos 4 de análisis previo; ajusta si necesitas)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    bank_df['cluster'] = kmeans.fit_predict(scaled_features)
    
    # PCA con instancia
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_features)
    pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
    pca_df['cluster'] = bank_df['cluster']
    
    return bank_df, scaler, kmeans, pca_df, pca  # Retornar instancia pca para transform

bank_df, scaler, kmeans, pca_df, pca = preprocess_and_cluster()

# Función para segmentar un nuevo cliente (interacción)
def predict_cluster(input_data):
    input_df = pd.DataFrame([input_data])
    
    # Codificar categóricas manualmente (mismo que en train)
    gender_map = {'Male': 0, 'Female': 1}
    region_map = {'North': 0, 'South': 1, 'East': 2, 'West': 3}
    channel_map = {'Online': 0, 'Branch': 1, 'Mobile': 2}
    
    input_df['gender'] = gender_map.get(input_df['gender'][0], 0)
    input_df['region'] = region_map.get(input_df['region'][0], 0)
    input_df['preferred_channel'] = channel_map.get(input_df['preferred_channel'][0], 0)
    
    # Escalar
    scaled_input = scaler.transform(input_df)
    
    # Predecir cluster
    cluster = kmeans.predict(scaled_input)[0]
    
    return cluster

# Menú en sidebar
st.sidebar.title("Menú")
section = st.sidebar.radio("Selecciona una Sección", [
    "Datos Históricos",
    "Análisis Exploratorio de Datos (EDA)",
    "Predicciones",
    "Segmentación de Clientes y Recomendaciones"
])

# Sección 1: Datos Históricos
if section == "Datos Históricos":
    st.title("Datos Históricos de Clientes")
    st.markdown("Visualización del dataset completo de clientes bancarios.")
    
    st.dataframe(bank_df)
    
    st.subheader("Resumen Estadístico")
    st.dataframe(bank_df.describe())

# Sección 2: Análisis Exploratorio de Datos (EDA)
elif section == "Análisis Exploratorio de Datos (EDA)":
    st.title("Análisis Exploratorio de Datos (EDA)")
    st.markdown("Exploración visual del dataset para entender patrones de clientes.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Matriz de Correlación")
        numeric_cols = numeric_cols = bank_df.select_dtypes(include=[np.number]).columns
        corr_matrix = bank_df[numeric_cols].corr()
        fig_corr = plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(fig_corr)
    
    with col2:
        st.subheader("Distribución de Edad por Género")
        fig_box = plt.figure(figsize=(6, 4))
        sns.boxplot(x='gender', y='age', data=bank_df, palette='viridis')
        st.pyplot(fig_box)
    
    st.subheader("Distribución de Ingreso Anual")
    fig_hist = plt.figure(figsize=(10, 6))
    sns.histplot(bank_df['annual_income'], bins=30, kde=True, color='#007bff')
    st.pyplot(fig_hist)

# Sección 3: Predicciones (Interacción con Usuario para Nuevo Cliente)
elif section == "Predicciones":
    st.title("Predicciones de Segmento para Nuevo Cliente")
    st.markdown("Ingrese datos de un nuevo cliente para predecir su cluster y obtener recomendaciones personalizadas.")
    
    # Formulario interactivo para inputs
    with st.form(key='customer_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Edad", 18, 80, 40)
            gender = st.selectbox("Género", ['Male', 'Female'])
            annual_income = st.number_input("Ingreso Anual (USD)", min_value=20000, max_value=200000, value=50000)
            account_balance = st.number_input("Saldo en Cuenta (USD)", min_value=0, max_value=500000, value=5000)
            #num_accounts = st.slider("Número de Cuentas", 1, 5, 2)
            num_transactions = st.slider("Transacciones Mensuales", 0, 100, 20)
            avg_transaction_amount = st.number_input("Monto Promedio Transacción (USD)", min_value=50, max_value=5000, value=200)
            credit_score = st.slider("Score Crediticio", 300, 850, 700)
        
        with col2:
  
            num_loans = st.selectbox("Tiene Préstamo", [0, 1])
            num_cards = st.selectbox("Tiene Tarjeta de Crédito", [0, 1])
            has_savings_account = st.selectbox("Cuenta de ahorros", [0, 1])
            has_investment = st.selectbox("Inversiones", [0, 1])
            loyalty_score = st.number_input("Puntuación de Lealtad (0-100)", min_value=0, max_value=100, value=50)
            region = st.selectbox("Región", ['North', 'South', 'East', 'West'])
            preferred_channel = st.selectbox("Canal Preferido", ['Online', 'Branch', 'Mobile'])
        
        submit = st.form_submit_button("Predecir Segmento")

    if submit:
        # Crear dict de input (sin customer_id)

        gender_map = {'Male': 0, 'Female': 1}
        region_map = {'North': 0, 'South': 1, 'East': 2, 'West': 3}
        channel_map = {'Online': 0, 'Branch': 1, 'Mobile': 2}

        input_data = {
            'age': age,
            'gender': gender_map[gender],   # convierte el texto a número
            'annual_income': annual_income,
            'account_balance': account_balance,
            'num_transactions': num_transactions,
            'avg_transaction_amount': avg_transaction_amount,
            'credit_score': credit_score,
            'num_loans': num_loans,
            'num_cards': num_cards,
            'has_savings_account': has_savings_account,
            'has_investment': has_investment,
            'loyalty_score': loyalty_score,
            'region': region_map[region],
            'preferred_channel': channel_map[preferred_channel],
        }
        
        # Predecir cluster
        cluster = predict_cluster(input_data)
        
        st.subheader("Resultados de Segmentación")
        st.markdown(f"**Cluster Asignado:** {cluster}")
        
        # Recomendación basada en cluster (de análisis previo)
        rec_map = {
            0: "Ofrecer tarjetas premium o inversiones (alto ingreso, buen crédito).",
            1: "Campañas de cashback en transacciones frecuentes (usuarios activos).",
            2: "Productos digitales/app banking para jóvenes (bajo saldo, alto uso mobile).",
            3: "Ofertas de préstamos básicos o educación financiera (bajo ingreso/crédito)."
        }
        st.markdown(f"**Recomendación Personalizada:** {rec_map.get(cluster, 'Análisis general: Revisar perfil manualmente.')}")
        
        # Visualización interactiva: Posición del nuevo cliente en PCA (aproximada)
        scaled_input = scaler.transform(pd.DataFrame([input_data]))
        pca_input = pca.transform(scaled_input)
        st.subheader("Posición del Nuevo Cliente en el Espacio de Clusters (PCA)")
        pca_df_new = pca_df.copy()
        # CORRECCIÓN: Usar pd.concat en lugar de append (deprecated en Pandas 2.0+)
        new_row = pd.DataFrame({'PC1': [pca_input[0][0]], 'PC2': [pca_input[0][1]], 'cluster': [cluster]})
        pca_df_new = pd.concat([pca_df_new, new_row], ignore_index=True)
        fig_pca = plt.figure(figsize=(10, 7))
        sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=pca_df_new, palette='viridis')
        plt.scatter(pca_input[0][0], pca_input[0][1], color='red', marker='X', s=200, label='Nuevo Cliente')
        plt.legend()
        st.pyplot(fig_pca)


# Sección 4: Segmentación de Clientes y Recomendaciones
else:
    st.title("Segmentación de Clientes y Recomendaciones")
    st.markdown("Visualización de clusters, perfiles y recomendaciones de marketing personalizadas.")
    
    st.subheader("Perfiles de Clusters (Promedios por Variable)")
    numeric_cols = numeric_cols = bank_df.select_dtypes(include=[np.number]).columns
    cluster_profiles = bank_df.groupby('cluster')[numeric_cols].mean().round(2)
    st.dataframe(cluster_profiles.style.background_gradient(cmap='viridis'))
    
    st.subheader("Recomendaciones por Cluster")
    rec_map = {
        0: "Ofrecer tarjetas premium o inversiones (alto ingreso, buen crédito).",
        1: "Campañas de cashback en transacciones frecuentes (usuarios activos).",
        2: "Productos digitales/app banking para jóvenes (bajo saldo, alto uso mobile).",
        3: "Ofertas de préstamos básicos o educación financiera (bajo ingreso/crédito)."
    }
    for cluster, rec in rec_map.items():
        st.markdown(f"**Cluster {cluster}:** {rec}")
    
    st.subheader("Visualización de Clusters con PCA")
    fig_pca = plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=pca_df, palette='viridis')
    st.pyplot(fig_pca)

# Pie de página
st.markdown("---")
st.caption("App desarrollada para segmentación de clientes bancarios. Datos sintéticos para demostración.")