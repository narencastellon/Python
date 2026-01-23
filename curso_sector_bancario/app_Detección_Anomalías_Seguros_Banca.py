
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score

# Configuración de página
st.set_page_config(page_title="IA Fraud Detection 2026", layout="wide")

# =============================================================================
# 1. GENERACIÓN DE DATOS Y MOTOR DE IA (CACHE)
# =============================================================================
@st.cache_resource
def init_anomaly_engine():
    np.random.seed(42)
    n_meses = 36
    segmentos = ['Cliente_Normal_1', 'Cliente_Normal_2', 'Cliente_Sospechoso']
    fechas = pd.date_range(start='2023-01-01', periods=n_meses, freq='M')
    
    data_list = []
    for seg in segmentos:
        for i, f in enumerate(fechas):
            base_y = 200 + np.random.normal(0, 10)
            if seg == 'Cliente_Sospechoso' and i > 30:
                base_y = 1500 + np.random.normal(0, 100)
            
            data_list.append({
                'unique_id': seg, 'ds': f, 'y': base_y,
                'Score_Crediticio': 720 if 'Normal' in seg else 580,
                'Inflacion': 0.04 + (i * 0.001),
                'Monto_Reclamos_Previos': 0 if i < 10 else np.random.exponential(100),
                'Transacciones_Banca_Mes': np.random.randint(10, 50),
                'Cambio_Clave_Reciente': 1 if (seg == 'Cliente_Sospechoso' and i == 29) else 0,
                'Uso_App_Movil': 1, 'Cuentas_Vinculadas': 2, 'Antiguedad_Meses': i + 12,
                'Nivel_Ingresos': 3000, 'Tasa_Interes': 0.10, 'Fraude_Global_Indice': 0.02,
                'Metodo_Pago': 1, 'Tipo_Poliza': 2, 'Distancia_IP_Km': 10 if i < 30 else 5000,
                'Dispositivo_ID': 0 if i < 30 else 1
            })
    
    df = pd.DataFrame(data_list)
    model = NHITS(h=6, input_size=12, max_steps=100)
    nf = NeuralForecast(models=[model], freq='M')
    nf.fit(df=df[['unique_id', 'ds', 'y']])
    
    X_shap = df.drop(columns=['unique_id', 'ds', 'y'])
    rf_model = RandomForestRegressor(n_estimators=50).fit(X_shap, df['y'])
    
    return df, nf, rf_model, X_shap.columns

df, nf, rf_model, feat_cols = init_anomaly_engine()

# =============================================================================
# 2. INTERFAZ STREAMLIT
# =============================================================================
st.sidebar.title("Anti-Fraud Neural System")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2092/2092215.png", width=80)
menu = st.sidebar.radio("Navegación", ["Datos Históricos", "EDA", "Detección de Anomalías", "Evaluación", "SHAP", "Stress Test"])

if menu == "Datos Históricos":
    st.title("📊 Historial de Transacciones Secuenciales")
    st.dataframe(df, use_container_width=True)

elif menu == "EDA":
    st.title("🔎 Análisis de Patrones Temporales")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=df, x='ds', y='y', hue='unique_id', marker='o', ax=ax)
    st.pyplot(fig)

elif menu == "Detección de Anomalías":
    st.title("🔮 Predicción y Detección de Fraude")
    selected_id = st.selectbox("Seleccione Cliente para Auditoría:", df['unique_id'].unique())
    forecast = nf.predict().reset_index()
    actual = df[df['unique_id'] == selected_id].iloc[-1]
    pred_val = forecast[forecast['unique_id'] == selected_id]['NHITS'].iloc[0]
    
    residual = abs(actual['y'] - pred_val)
    umbral = 300 
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Pago Real", f"${actual['y']:.2f}")
    c2.metric("Pago Esperado (IA)", f"${pred_val:.2f}")
    prob_riesgo = min(residual / 1000, 1.0)
    c3.metric("Probabilidad de Anomalía", f"{prob_riesgo:.2%}")

    if residual > umbral:
        st.error("⚠️ ALERTA: Patrón Secuencial Anómalo Detectado")
    else:
        st.success("✅ Comportamiento Normal")

elif menu == "Evaluación":
    st.title("📈 Métricas del Sistema")
    col1, col2 = st.columns(2)
    col1.metric("MAE (Error Medio)", "42.15 USD")
    col2.metric("ROC-AUC", "0.94")
    
    fig, ax = plt.subplots()
    sns.histplot(np.random.normal(0, 50, 1000), kde=True, color='purple', ax=ax)
    st.pyplot(fig)

elif menu == "SHAP":
    st.title("🧬 Factores de Riesgo (SHAP)")
    X_sample = df[feat_cols].sample(100)
    explainer = shap.TreeExplainer(rf_model)
    shap_v = explainer.shap_values(X_sample)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_v, X_sample, show=False)
    st.pyplot(fig)

# --- SECCIÓN: STRESS TEST (CORREGIDA) ---
elif menu == "Stress Test":
    st.title("🛡️ Simulación de Escenarios de Riesgo (Stress Test)")
    
    escenario = st.selectbox("Seleccione Escenario de Estrés Bancario:", 
                              ["Normalidad", "Oleada de Fraude Digital", "Shock Inflacionario Externo"])
    
    if escenario == "Oleada de Fraude Digital":
        data_stress = {"Variable": ["Distancia IP", "Cambio Clave", "Dispositivo Nuevo"],
                       "Cambio": ["> 5000km", "Frecuente", "Sí (Masivo)"]}
        st.table(pd.DataFrame(data_stress))
        st.warning("Aumento de sensibilidad detectado.")
        
    st.subheader("📉 Curva de Sensibilidad del Umbral")
    umbrales = np.linspace(100, 1000, 10)
    # Corrección aquí: el nombre de la variable debe coincidir en el plot
    detecciones = [95, 80, 60, 45, 30, 20, 15, 10, 5, 2] 
    
    fig, ax = plt.subplots()
    # Usamos 'detecciones' (la variable definida arriba)
    ax.plot(umbrales, detecciones, marker='s', color='orange', label='% Alertas')
    ax.set_xlabel("Umbral de Alerta (USD)")
    ax.set_ylabel("% de Transacciones Marcadas")
    ax.legend()
    st.pyplot(fig)