import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split

from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve)
import shap

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Credit Risk Sentinel", layout="wide")

# --- 2. MOTOR DE DATOS Y ENTRENAMIENTO (15 VARIABLES) ---
@st.cache_resource
def load_system():
    np.random.seed(42)
    n = 2500
    # Definición de las 15 variables profesionales
    data = {
        'Edad': np.random.randint(18, 75, n),
        'Ingresos_Anuales': np.random.gamma(shape=5, scale=12000, size=n),
        'Puntaje_Credito': np.random.randint(300, 850, n),
        'Monto_Prestamo': np.random.gamma(shape=3, scale=6000, size=n),
        'Tasa_Interes': np.random.uniform(0.05, 0.30, n),
        'Ratio_Deuda_Ingreso': np.random.uniform(0.1, 0.7, n),
        'Antiguedad_Empleo': np.random.randint(0, 360, n),
        'Num_Tarjetas': np.random.randint(0, 10, n),
        'Uso_Linea_Credito': np.random.uniform(0.0, 1.0, n),
        'Ahorros_Liquidos': np.random.gamma(shape=2, scale=4000, size=n),
        'Historial_Mora_6m': np.random.choice([0, 1, 2], n, p=[0.8, 0.15, 0.05]),
        'Educacion': np.random.choice([0, 1, 2], n),
        'Propiedad_Vivienda': np.random.choice([0, 1], n),
        'Num_Dependientes': np.random.randint(0, 5, n),
        'Gastos_Mensuales': np.random.normal(2500, 600, n)
    }
    df = pd.DataFrame(data)
    
    # Lógica de Target (Morosidad)
    logit = (df['Ratio_Deuda_Ingreso'] * 4 + df['Uso_Linea_Credito'] * 3 + 
             df['Historial_Mora_6m'] * 2.5 - (df['Puntaje_Credito'] / 150))
    prob = 1 / (1 + np.exp(-logit + 4))
    df['Morosidad'] = (prob > np.random.uniform(0, 1, n)).astype(int)
    
    # Entrenamiento
    X = df.drop('Morosidad', axis=1)
    y = df['Morosidad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    return df, model, X_test, y_test

df, model, X_test, y_test = load_system()

# --- 3. MENÚ LATERAL ---
st.sidebar.title("💳 Credit Sentinel AI")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
menu = st.sidebar.radio("Navegación", 
    ["Datos Históricos", "EDA", "Predicción de Riesgo", "Evaluación Modelo", "Explicabilidad SHAP", "Monte Carlo & Estrés"])

# --- MODULO 1: DATOS HISTÓRICOS ---
if menu == "Datos Históricos":
    st.title("📋 Registro de Operaciones")
    st.write("Base de datos histórica con 15 variables clave de riesgo.")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Resumen Estadisticos")
    st.dataframe(df.describe().T)

# --- MODULO 2: EDA ---
elif menu == "EDA":
    st.title("🔎 Análisis Exploratorio")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Puntaje de Crédito vs Morosidad")
        fig1, ax1 = plt.subplots()
        sns.boxplot(x='Morosidad', y='Puntaje_Credito', data=df, ax=ax1, palette="RdYlGn_r")
        st.pyplot(fig1)
    with col2:
        st.subheader("Relación Deuda/Ingreso")
        fig2, ax2 = plt.subplots()
        sns.kdeplot(data=df, x="Ratio_Deuda_Ingreso", hue = "Morosidad", fill=True, ax=ax2)
        st.pyplot(fig2)
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Correlación de Variables")
        fig3, ax3 = plt.subplots()
        sns.heatmap(df.corr(), cmap='RdYlGn', ax=ax3)
        st.pyplot(fig3)
    with col4:
        st.subheader("Default por Historial de Mora")
        fig4, ax4 = plt.subplots()
        sns.countplot(x= 'Historial_Mora_6m', hue = "Morosidad", data=df, ax=ax4)
        st.pyplot(fig4)

# --- MODULO 3: PREDICCIÓN (SOLUCIÓN AL ERROR DE COLUMNAS) ---
elif menu == "Predicción de Riesgo":
    st.title("🔮 Evaluación de Crédito Individual")
    
    with st.form("loan_form"):
        st.subheader("Datos del Perfil")
        c1, c2, c3 = st.columns(3)
        
        # Inputs dinámicos
        puntaje = c1.slider("Puntaje Crédito", 300, 850, 650)
        ingresos = c2.number_input("Ingresos Anuales ($)", 5000, 250000, 50000)
        dti = c3.slider("Ratio Deuda/Ingreso", 0.0, 1.0, 0.3)
        uso_linea = c1.slider("Uso Línea Crédito (%)", 0.0, 1.0, 0.4)
        mora = c2.selectbox("Mora previa (6 meses)", [0, 1, 2])
        monto = c3.number_input("Monto del Préstamo", 1000, 100000, 20000)
        tasa = c1.slider("Tasa Interés Aplicada", 0.05, 0.40, 0.15)
        
        submitted = st.form_submit_button("🚀 Calcular Riesgo")

    if submitted:
        # CONSTRUCCIÓN ROBUSTA: Creamos el diccionario con todas las 15 columnas
        perfil = {col: df[col].median() for col in X_test.columns} # Valores base
        perfil.update({
            'Puntaje_Credito': puntaje,
            'Ingresos_Anuales': ingresos,
            'Ratio_Deuda_Ingreso': dti,
            'Uso_Linea_Credito': uso_linea,
            'Historial_Mora_6m': mora,
            'Monto_Prestamo': monto,
            'Tasa_Interes': tasa
        })
        
        # Convertir a DataFrame asegurando el orden correcto de las 15 columnas
        input_df = pd.DataFrame([perfil])[X_test.columns]
        
        prob = model.predict_proba(input_df)[0, 1]
        
        st.divider()
        res1, res2 = st.columns(2)
        with res1:
            st.metric("Probabilidad de Mora", f"{prob:.2%}")
            if prob < 0.25:
                st.success("RIESGO: BAJO")
            elif prob < 0.60:
                st.warning("RIESGO: MEDIO")
            else:
                st.error("RIESGO: ALTO")
        with res2:
            st.write("Resumen Ejecutivo:")
            st.write(f"El cliente presenta un perfil de riesgo basado en un DTI de {dti:.0%} y un score de {puntaje}.")

# --- MODULO 4: EVALUACIÓN MODELO ---
elif menu == "Evaluación Modelo":
    st.title("📈 Métricas de Performance")
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = model.predict(X_test)
    
    c1, c2 = st.columns(2)
    with c1:
        st.text("Matriz de Confusión")
        fig1, ax1 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_preds), annot=True, fmt='d', cmap='Greens')
        st.pyplot(fig1)
    with c2:
        st.metric("ROC-AUC Score", f"{roc_auc_score(y_test, y_probs):.4f}")
        st.text("Reporte de Clasificación:")
        st.code(classification_report(y_test, y_preds))

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Curva ROC-AUC")
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        fig3, ax3 = plt.subplots()
        ax3.plot(fpr, tpr, label=f'AUC: {roc_auc_score(y_test, y_probs):.4f}')
        ax3.plot([0,1], [0,1], 'k--')
        plt.legend()
        st.pyplot(fig3)

# --- MODULO 5: SHAP ---
elif menu == "Explicabilidad SHAP":
    st.title("🧬 ¿Por qué el modelo toma estas decisiones?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig)

# --- MODULO 6: MONTE CARLO & ESTRÉS ---
elif menu == "Monte Carlo & Estrés":
    st.title("🎲 Análisis Estocástico de Monte Carlo")
    
    st.subheader("Simulación de Escenario de Recesión")
    col_s1, col_s2 = st.columns(2)
    caida_ingreso = col_s1.slider("Caída Ingresos (%)", 0, 50, 20) / 100
    alza_uso = col_s2.slider("Alza Uso Línea (%)", 0, 50, 30) / 100
    
    df_s = X_test.copy()
    df_s['Ingresos_Anuales'] *= (1 - caida_ingreso)
    df_s['Uso_Linea_Credito'] = np.clip(df_s['Uso_Linea_Credito'] * (1 + alza_uso), 0, 1)
    
    p_base = model.predict_proba(X_test)[:, 1].mean()
    p_stress = model.predict_proba(df_s)[:, 1].mean()
    
    m1, m2 = st.columns(2)
    m1.metric("Default Base", f"{p_base:.2%}")
    m2.metric("Default Bajo Estrés", f"{p_stress:.2%}", f"+{p_stress-p_base:.2%}")
    
    st.subheader("Curva de Sensibilidad: Ingresos vs Default")
    impacto = []
    rango = np.linspace(0, 0.6, 10)
    for r in rango:
        df_temp = X_test.copy()
        df_temp['Ingresos_Anuales'] *= (1 - r)
        impacto.append(model.predict_proba(df_temp)[:, 1].mean())
    
    fig, ax = plt.subplots()
    ax.plot(rango*100, impacto, marker='o', color='red')
    ax.set_xlabel("% de reducción de ingresos")
    ax.set_ylabel("Tasa de Default de Cartera")
    st.pyplot(fig)