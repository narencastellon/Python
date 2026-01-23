import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Configuración de la página
st.set_page_config(page_title="Banking Cross-Selling App", layout="wide")

# =============================================================================
# 1. GENERACIÓN DE DATOS Y MODELADO (CACHED)
# =============================================================================
@st.cache_resource
def load_data_and_models():
    # Simulación de datos (15 variables)
    np.random.seed(42)
    n_samples = 2000
    data = {
        'Edad': np.random.randint(18, 75, n_samples),
        'Genero': np.random.choice(['M', 'F'], n_samples),
        'Estado_Civil': np.random.choice(['Soltero', 'Casado', 'Divorciado'], n_samples),
        'Nivel_Educacion': np.random.choice(['Secundaria', 'Universitario', 'Postgrado'], n_samples),
        'Ubicacion': np.random.choice(['Urbano', 'Rural', 'Suburbano'], n_samples),
        'Ingreso_Anual': np.random.normal(50000, 15000, n_samples),
        'Score_Crediticio': np.random.normal(650, 100, n_samples),
        'Saldo_Cuenta': np.random.normal(12000, 5000, n_samples),
        'Tenencia_Anios': np.random.randint(1, 20, n_samples),
        'Num_Productos': np.random.randint(1, 5, n_samples),
        'Tiene_TC': np.random.choice([0, 1], n_samples),
        'Miembro_Activo': np.random.choice([0, 1], n_samples),
        'Transacciones_Mes': np.random.randint(0, 50, n_samples),
        'Reclamos_Previos': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Uso_App_Movil': np.random.choice([0, 1], n_samples)
    }
    df = pd.DataFrame(data)
    
    # Target sintético
    logit = (df['Edad'] * 0.02) + (df['Ingreso_Anual'] * 0.00002) + (df['Score_Crediticio'] * 0.003) - 5
    prob = 1 / (1 + np.exp(-logit))
    df['Target_CrossSell'] = np.random.binomial(1, prob)
    
    # Preprocesamiento
    df_proc = df.copy()
    le_dict = {}
    for col in ['Genero', 'Estado_Civil', 'Nivel_Educacion', 'Ubicacion']:
        le = LabelEncoder()
        df_proc[col] = le.fit_transform(df_proc[col])
        le_dict[col] = le
        
    features = df_proc.drop(columns=['Target_CrossSell'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # GMM Clustering
    gmm = GaussianMixture(n_components=3, random_state=42)
    df['Segmento'] = gmm.fit_predict(X_scaled)
    
    # Regresión Logística
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Target_CrossSell'], test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return df, df_proc, model, gmm, scaler, features.columns, le_dict

df, df_proc, model, gmm, scaler, feat_cols, le_dict = load_data_and_models()

# =============================================================================
# 2. INTERFAZ DE STREAMLIT (MENÚ)
# =============================================================================
st.sidebar.title("Naren Castellon - Forecasting 2026")
# 
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
menu = st.sidebar.radio("Navegación", ["Históricos", "EDA", "Predicción Individual", "Evaluación Modelo", "SHAP Analysis", "Análisis de Riesgo", "Sensibilidad Robust"])

# --- SECCIÓN: HISTÓRICOS ---
if menu == "Históricos":
    st.title("📊 Datos Históricos de Clientes")
    st.dataframe(df.head(50))
    st.download_button("Descargar Dataset", df.to_csv().encode('utf-8'), "data_banca.csv")

# --- SECCIÓN: EDA ---
elif menu == "EDA":
    st.title("🔎 Análisis Exploratorio de Datos")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución de Segmentos (GMM)")
        fig, ax = plt.subplots()
        df['Segmento'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=['#ff9999','#66b3ff','#99ff99'])
        st.pyplot(fig)
    with col2:
        st.subheader("Ingreso vs Score por Segmento")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='Ingreso_Anual', y='Score_Crediticio', hue='Segmento', ax=ax)
        st.pyplot(fig)

# --- SECCIÓN: PREDICCIÓN INDIVIDUAL ---
elif menu == "Predicción Individual":
    st.title("🔮 Predicción de Cross-Selling")
    st.subheader("Ingrese los datos del cliente:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        edad = st.number_input("Edad", 18, 90, 35)
        ingreso = st.number_input("Ingreso Anual", 10000, 200000, 50000)
        score = st.slider("Score Crediticio", 300, 850, 650)
    with col2:
        genero = st.selectbox("Género", le_dict['Genero'].classes_)
        educacion = st.selectbox("Educación", le_dict['Nivel_Educacion'].classes_)
        civil = st.selectbox("Estado Civil", le_dict['Estado_Civil'].classes_)
    with col3:
        ubicacion = st.selectbox("Ubicación", le_dict['Ubicacion'].classes_)
        transacciones = st.number_input("Transacciones/Mes", 0, 100, 20)
        reclamos = st.radio("¿Tiene Reclamos Previos?", [0, 1])

    # Procesar entrada
    input_data = pd.DataFrame([[edad, genero, civil, educacion, ubicacion, ingreso, score, 12000, 5, 2, 1, 1, transacciones, reclamos, 1]], columns=feat_cols)
    for col in le_dict:
        input_data[col] = le_dict[col].transform(input_data[col])
    
    input_scaled = scaler.transform(input_data)
    
    if st.button("Generar Diagnóstico"):
        prob_venta = model.predict_proba(input_scaled)[0][1]
        segmento_pred = gmm.predict(input_scaled)[0]
        
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilidad de Compra", f"{prob_venta:.2%}")
        c2.metric("Segmento Asignado", f"Cluster {segmento_pred}")
        
        # Lógica de Riesgo
        riesgo = "Alto" if score < 600 else "Bajo" if score > 750 else "Medio"
        c3.metric("Nivel de Riesgo", riesgo)
        
        st.success(f"Recomendación: {'Ofrecer Seguro Premium' if prob_venta > 0.6 else 'Seguro Básico o Mantener Observación'}")

# --- SECCIÓN: EVALUACIÓN ---
elif menu == "Evaluación Modelo":
    st.title("📈 Evaluación de Performance")
    # Cálculos rápidos
    y_test_prob = model.predict_proba(scaler.transform(df_proc.drop(columns='Target_CrossSell')))[:, 1]
    
    col1, col2 = st.columns(2)
    with col1:
        st.text("Reporte de Clasificación:")
        st.code(classification_report(df['Target_CrossSell'], model.predict(scaler.transform(df_proc.drop(columns='Target_CrossSell')))))
    with col2:
        fig, ax = plt.subplots()
        fpr, tpr, _ = roc_curve(df['Target_CrossSell'], y_test_prob)
        ax.plot(fpr, tpr, label=f"AUC: {roc_auc_score(df['Target_CrossSell'], y_test_prob):.2f}")
        ax.plot([0,1],[0,1], 'k--')
        plt.legend()
        st.pyplot(fig)

# --- SECCIÓN: SHAP ---
elif menu == "SHAP Analysis":
    st.title("🧬 Explicabilidad con SHAP")
    explainer = shap.LinearExplainer(model, scaler.transform(df_proc.drop(columns='Target_CrossSell')))
    shap_values = explainer.shap_values(scaler.transform(df_proc.drop(columns='Target_CrossSell')))
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, scaler.transform(df_proc.drop(columns='Target_CrossSell')), feature_names=feat_cols, show=False)
    st.pyplot(fig)

# --- SECCIÓN: ANÁLISIS DE RIESGO ---
elif menu == "Análisis de Riesgo":
    st.title("⚠️ Matriz de Riesgo Bancario")
    df_risk = df.copy()
    df_risk['Prob_Compra'] = model.predict_proba(scaler.transform(df_proc.drop(columns='Target_CrossSell')))[:, 1]
    
    conditions = [(df_risk['Score_Crediticio'] < 600), (df_risk['Score_Crediticio'] < 750), (df_risk['Score_Crediticio'] >= 750)]
    choices = ['Riesgo Alto', 'Riesgo Medio', 'Riesgo Bajo']
    df_risk['Nivel_Riesgo'] = np.select(conditions, choices, default='N/A')
    
    st.write(df_risk[['Edad', 'Ingreso_Anual', 'Score_Crediticio', 'Nivel_Riesgo', 'Prob_Compra']].head(20))
    
    fig, ax = plt.subplots()
    sns.boxplot(x='Nivel_Riesgo', y='Prob_Compra', data=df_risk, ax=ax)
    st.pyplot(fig)

# --- SECCIÓN: SENSIBILIDAD ---
elif menu == "Sensibilidad Robust":
    st.title("🛡️ Análisis de Sensibilidad (Stress Test)")
    st.write("¿Cómo varía la probabilidad de venta ante cambios en el Score Crediticio?")
    
    base_client = scaler.transform(df_proc.drop(columns='Target_CrossSell')).mean(axis=0).reshape(1, -1)
    score_idx = list(feat_cols).index('Score_Crediticio')
    
    scores_sim = np.linspace(-2, 2, 20) # Valores estandarizados
    probs = []
    for s in scores_sim:
        base_client[0, score_idx] = s
        probs.append(model.predict_proba(base_client)[0][1])
        
    fig, ax = plt.subplots()
    ax.plot(scores_sim, probs, marker='o', color='red')
    ax.set_xlabel("Score Crediticio (Estandarizado)")
    ax.set_ylabel("Probabilidad de Venta")
    st.pyplot(fig)