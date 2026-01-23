# =======================================================
# App Streamlit Dinámica: Forecasting de Volúmenes de Ventas de Seguros en Canales Bancarios
# Usa NeuralForecast (NHITS) para forecasting de pólizas vendidas con covariables externas
# Simula escenarios dinámicos (base/stress) con sliders + Monte Carlo interactivo para impacto en ingresos/cartera/reservas
# Incluye: Imagen header, menú lateral, datos históricos, EDA, predicciones (forecast interactivo, impacto ventas, riesgo cliente personalizado),
# evaluación del modelo, análisis SHAP, medición de riesgo clientes, sensibilidad robusta y Monte Carlo
# =======================================================
# Requisitos: pip install streamlit pandas numpy scikit-learn matplotlib seaborn neuralforecast shap

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE
from neuralforecast.losses.numpy import mae, rmse
import shap
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(page_title="Forecasting Ventas Seguros Bancarios - Naren Castellón", layout="wide")
st.title("📊 Forecasting de Volúmenes de Ventas de Seguros en Canales Bancarios")
st.markdown("**NeuralForecast + Monte Carlo Dinámico para escenarios y riesgos** | Dictado por Naren Castellón")

# Imagen header
st.sidebar.image("https://images.unsplash.com/photo-1554224155-6726b3ff858f?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80", 
         caption="Forecasting de ventas de seguros en canales bancarios con series temporales e IA",)

# =======================================================
# Carga de datos y modelos (cacheado)
# =======================================================
@st.cache_resource
def load_data_and_models():
    np.random.seed(42)
    dates = pd.date_range('2015-01-01', '2025-12-01', freq='M')
    n_periods = len(dates)

    macro_data = {
        'gdp_growth': np.cumsum(np.random.normal(0.2, 1, n_periods)) + 3,
        'unemployment_rate': np.abs(np.cumsum(np.random.normal(0, 0.3, n_periods)) + 5),
        'inflation_rate': np.abs(np.random.normal(2, 1, n_periods)),
        'interest_rate': np.abs(np.random.normal(3, 1.5, n_periods)).clip(0.5, 10),
        'consumer_confidence': np.random.normal(100, 15, n_periods).clip(50, 150),
        'digital_adoption_index': np.cumsum(np.random.normal(0.5, 2, n_periods)) + 50,
        'bank_branch_density': np.random.normal(20, 5, n_periods).clip(10, 40),
        'marketing_spend': np.random.normal(500000, 100000, n_periods).clip(200000, 1000000)
    }

    bank_sales_data = {
        'active_bank_clients': np.cumsum(np.random.normal(1000, 200, n_periods)) + 500000,
        'cross_sell_rate': np.random.normal(15, 3, n_periods).clip(5, 30),
        'policies_sold': np.abs(np.random.normal(5000, 1500, n_periods) + 
                               200 * macro_data['gdp_growth'] + 
                               100 * macro_data['consumer_confidence'] + 
                               50 * macro_data['digital_adoption_index'] + 
                               0.001 * macro_data['marketing_spend']).astype(int),
        'average_premium': np.random.normal(1200, 300, n_periods).clip(500, 3000),
        'claim_rate': np.abs(np.random.normal(8, 2, n_periods)),
        'customer_acquisition_cost': np.random.normal(200, 50, n_periods).clip(100, 400),
        'retention_rate': np.random.normal(85, 5, n_periods).clip(70, 95),
        'net_promoter_score': np.random.normal(40, 10, n_periods).clip(10, 70)
    }

    df_aggregate = pd.DataFrame({**macro_data, **bank_sales_data})
    df_aggregate['ds'] = dates

    # Clientes individuales
    n_clients = 10000
    client_data = {
        'client_id': range(1, n_clients + 1),
        'age': np.random.randint(25, 70, n_clients),
        'income': np.random.normal(60000, 25000, n_clients).clip(20000, 150000).astype(int),
        'credit_score': np.random.normal(700, 100, n_clients).clip(300, 850).astype(int),
        'debt_to_income_ratio': np.round(np.random.uniform(0.1, 0.6, n_clients), 2),
        'num_bank_products': np.random.randint(1, 6, n_clients),
        'tenure_years': np.random.randint(1, 20, n_clients),
        'digital_usage_score': np.random.randint(1, 10, n_clients),
        'satisfaction_score': np.random.randint(1, 10, n_clients),
        'previous_insurance': np.random.choice([0, 1], n_clients, p=[0.7, 0.3]),
        'family_size': np.random.randint(1, 6, n_clients),
        'employment_stability': np.random.choice(['Stable', 'Unstable'], n_clients),
        'region': np.random.choice(['Urban', 'Suburban', 'Rural'], n_clients),
        'marketing_exposure': np.random.randint(0, 5, n_clients),
        'complaints': np.random.randint(0, 3, n_clients),
        'life_events': np.random.choice(['None', 'Marriage', 'Child Birth', 'Home Purchase'], n_clients)
    }

    df_clients = pd.DataFrame(client_data)

    buy_prob_client = (
        0.20 + 
        0.15 * (df_clients['income'] > 80000) + 
        0.12 * (df_clients['credit_score'] > 700) + 
        0.10 * (df_clients['satisfaction_score'] > 7) + 
        0.08 * (df_clients['num_bank_products'] > 3) + 
        0.07 * (df_clients['digital_usage_score'] > 7) + 
        0.05 * (df_clients['previous_insurance'] == 1) + 
        -0.10 * (df_clients['complaints'] > 0)
    )
    buy_prob_client = buy_prob_client.clip(0, 0.95)
    df_clients['buy_insurance'] = np.random.binomial(1, buy_prob_client)

    # Preprocesamiento clientes para propensión
    cat_cols = df_clients.select_dtypes(include='object').columns.drop('client_id', errors='ignore')
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_clients[col] = le.fit_transform(df_clients[col])
        encoders[col] = le

    X_client = df_clients.drop(['client_id', 'buy_insurance'], axis=1)
    y_client = df_clients['buy_insurance']
    scaler_client = StandardScaler()
    X_client_scaled = scaler_client.fit_transform(X_client)

    model_prop = LogisticRegression(random_state=42)
    model_prop.fit(X_client_scaled, y_client)

    # Forecasting
    exog_cols = ['consumer_confidence', 'gdp_growth', 'marketing_spend', 'unemployment_rate']
    df_ts = pd.DataFrame({
        'unique_id': ['policies_sold'] * n_periods,
        'ds': dates,
        'y': df_aggregate['policies_sold']
    })
    for col in exog_cols:
        df_ts[col] = df_aggregate[col]

    train_df = df_ts[df_ts['ds'] < '2024-01-01']

    future_dates = pd.date_range('2024-01-01', '2025-12-01', freq='M')
    futr_df_base = pd.DataFrame({
        'unique_id': ['policies_sold'] * len(future_dates),
        'ds': future_dates
    })
    for col in exog_cols:
        futr_df_base[col] = df_aggregate[col].iloc[-1]

    horizon = len(future_dates)

    models = [NHITS(h=horizon, input_size=36, futr_exog_list=exog_cols, loss=MAE(), max_steps=200, random_seed=42)]
    nf = NeuralForecast(models=models, freq='M')
    nf.fit(df=train_df)
    forecast_base = nf.predict(futr_df=futr_df_base)

    # SHAP para propensión cliente
    background_shap = shap.kmeans(X_client_scaled, 50).data
    explainer_prop = shap.KernelExplainer(model_prop.predict_proba, background_shap)
    shap_values_prop = explainer_prop.shap_values(X_client_scaled[:100])

    training_columns = X_client.columns.tolist()  # Guardar orden columnas entrenamiento

    return (df_aggregate, df_clients, forecast_base, model_prop, scaler_client, encoders, X_client_scaled,
            explainer_prop, shap_values_prop, nf, exog_cols, futr_df_base, training_columns)

(df_aggregate, df_clients, forecast_base, model_prop, scaler_client, encoders, X_client_scaled,
 explainer_prop, shap_values_prop, nf, exog_cols, futr_df_base, training_columns) = load_data_and_models()

# =======================================================
# Menú lateral
# =======================================================
st.sidebar.title("Navegación")
section = st.sidebar.radio("Secciones", 
                           ["Datos Históricos", "EDA", "Predicciones", 
                            "Evaluación del Modelo", "Análisis SHAP", 
                            "Medición de Riesgo de Clientes", "Análisis de Sensibilidad Robusta",
                            "Simulación de Monte Carlo"])

# =======================================================
# Secciones
# =======================================================
if section == "Datos Históricos":
    st.header("📊 Datos Históricos")
    st.write("Datos Agregados (Series Temporales)")
    st.dataframe(df_aggregate.tail(50))
    st.write("Datos Clientes Individuales (muestra)")
    st.dataframe(df_clients.head(200))

elif section == "EDA":
    st.header("🔍 Análisis Exploratorio de Datos (EDA)")
    st.write("Series Temporales Ventas Pólizas")
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(df_aggregate['ds'], df_aggregate['policies_sold'], label='Pólizas Vendidas')
    ax.plot(df_aggregate['ds'], df_aggregate['consumer_confidence'], label='Confianza Consumidor', alpha=0.7)
    ax.plot(df_aggregate['ds'], df_aggregate['marketing_spend']/1000, label='Marketing Spend (miles)', alpha=0.7)
    ax.legend()
    st.pyplot(fig)

    st.write("Distribución Compra Seguro Clientes")
    fig, ax = plt.subplots()
    sns.countplot(x='buy_insurance', data=df_clients, ax=ax)
    st.pyplot(fig)

elif section == "Predicciones":
    st.header("🔮 Predicciones")
    st.write("Escenarios Forecast Ventas Pólizas (Base vs Stress Interactivo)")
    confidence_shock = st.slider("Shock Confianza Consumidor (%)", -30.0, 20.0, -20.0)
    marketing_shock = st.slider("Shock Marketing Spend (%)", -50.0, 50.0, -30.0)
    unemployment_shock = st.slider("Shock Desempleo (%)", 0.0, 10.0, 3.0)

    futr_df_stress = futr_df_base.copy()
    futr_df_stress['consumer_confidence'] *= (1 + confidence_shock/100)
    futr_df_stress['marketing_spend'] *= (1 + marketing_shock/100)
    futr_df_stress['unemployment_rate'] *= (1 + unemployment_shock/10)

    forecast_stress_dynamic = nf.predict(futr_df=futr_df_stress)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Forecast Pólizas Vendidas")
        fig, ax = plt.subplots()
        base_sales = forecast_base['NHITS']
        stress_sales = forecast_stress_dynamic['NHITS']
        ax.plot(base_sales.index, base_sales, label='Base')
        ax.plot(stress_sales.index, stress_sales, label='Stress')
        ax.legend()
        st.pyplot(fig)
    with col2:
        st.write("Diferencia Ventas (Stress - Base)")
        diff = stress_sales - base_sales
        fig, ax = plt.subplots()
        ax.bar(diff.index, diff)
        st.pyplot(fig)

    st.write("Impacto en Ingresos Seguros")
    average_premium = df_aggregate['average_premium'].mean()
    base_revenue = base_sales.mean() * average_premium * 12
    stress_revenue = stress_sales.mean() * average_premium * 12
    st.metric("Ingresos Anuales Base", f"${base_revenue:,.0f}")
    st.metric("Ingresos Anuales Stress", f"${stress_revenue:,.0f}", delta=f"${stress_revenue - base_revenue:,.0f}")

    st.write("Ejemplo Cliente Individual (Propensión Compra bajo Stress)")
    with st.form("cliente_ejemplo"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Edad", 25, 70, 40)
            income = st.number_input("Ingreso Anual ($)", 20000, 150000, 60000)
            credit_score = st.number_input("Credit Score", 300, 850, 700)
            debt_to_income_ratio = st.number_input("DTI", 0.1, 0.6, 0.3)
            num_bank_products = st.number_input("Número Productos Bancarios", 1, 6, 3)
            tenure_years = st.number_input("Años Cliente", 1, 20, 5)
            digital_usage_score = st.number_input("Uso Digital (1-10)", 1, 10, 7)
            satisfaction_score = st.number_input("Satisfacción (1-10)", 1, 10, 8)
        with col2:
            previous_insurance = st.selectbox("Seguro Previo", [0, 1])
            complaints = st.number_input("Quejas Último Año", 0, 3, 0)
            employment_stability = st.selectbox("Estabilidad Empleo", ['Stable', 'Unstable'])
            region = st.selectbox("Región", ['Urban', 'Suburban', 'Rural'])
            marketing_exposure = st.number_input("Exposición Marketing", 0, 5, 3)
            life_events = st.selectbox("Eventos Vida", ['None', 'Marriage', 'Child Birth', 'Home Purchase'])
            family_size = st.number_input("Tamaño Familia", 1, 6, 2)

        submitted = st.form_submit_button("Calcular Propensión Cliente")

    if submitted:
        input_client = pd.DataFrame({
            'age': [age], 'income': [income], 'credit_score': [credit_score],
            'debt_to_income_ratio': [debt_to_income_ratio], 'num_bank_products': [num_bank_products],
            'tenure_years': [tenure_years], 'digital_usage_score': [digital_usage_score],
            'satisfaction_score': [satisfaction_score], 'previous_insurance': [previous_insurance],
            'family_size': [family_size], 'employment_stability': [employment_stability],
            'region': [region], 'marketing_exposure': [marketing_exposure],
            'complaints': [complaints], 'life_events': [life_events]
        })

        # Encoding categóricas
        for col, le in encoders.items():
            if col in input_client.columns:
                input_client[col] = le.transform(input_client[col])

        # Reindex para coincidir exactamente con columnas entrenamiento
        input_client = input_client.reindex(columns=training_columns, fill_value=0)

        input_scaled = scaler_client.transform(input_client)

        prob_base = model_prop.predict_proba(input_scaled)[0][1]
        stress_factor = forecast_stress_dynamic['NHITS'].mean() / forecast_base['NHITS'].mean()
        prob_stress = np.clip(prob_base * stress_factor, 0, 1)
        risk = "Bajo" if prob_stress > 0.7 else "Medio" if prob_stress > 0.3 else "Alto"

        st.metric("Propensión Base", f"{prob_base:.2%}")
        st.metric("Propensión Stress", f"{prob_stress:.2%}")
        st.metric("Nivel de Riesgo (no compra)", risk)

        st.subheader("Perfil del Cliente")
        st.table(input_client.T)


elif section == "Evaluación del Modelo":
    st.header("📈 Evaluación del Modelo")
    st.subheader("Métricas Forecast NHITS (estilo Power BI)")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE Forecast", "Ejemplo: 500")
    col2.metric("RMSE Forecast", "Ejemplo: 700")
    col3.metric("Horizonte (meses)", 24)

    st.subheader("Métricas Propensión Cliente (estilo Power BI)")
    auc_prop = roc_auc_score(df_clients['buy_insurance'], model_prop.predict_proba(X_client_scaled)[:,1])
    col1, col2, col3 = st.columns(3)
    col1.metric("AUC-ROC Propensión", f"{auc_prop:.4f}")
    col2.metric("Accuracy Propensión", "Ejemplo: 0.85")
    col3.metric("Precision Propensión", "Ejemplo: 0.78")

    st.write("Classification Report Propensión Cliente")
    y_pred_prop = model_prop.predict(X_client_scaled)
    st.code(classification_report(df_clients['buy_insurance'], y_pred_prop))

elif section == "Análisis SHAP":
    st.header("🧠 Análisis de Explicabilidad con SHAP")
    st.write("Importancia Global Propensión Compra")
    shap.summary_plot(shap_values_prop[1], X_client_scaled[:100], feature_names=df_clients.drop(['client_id', 'buy_insurance'], axis=1).columns, show=False)
    st.pyplot(plt.gcf())

elif section == "Medición de Riesgo de Clientes":
    st.header("⚠️ Medición de Riesgo de Clientes")
    confidence_shock_risk = st.slider("Shock Confianza Consumidor para Riesgo (%)", -30.0, 20.0, -20.0, key="risk_conf")
    marketing_shock_risk = st.slider("Shock Marketing para Riesgo (%)", -50.0, 50.0, -30.0, key="risk_mark")
    stress_factor_risk = forecast_base['NHITS'].mean() * (1 + confidence_shock_risk/100) * (1 + marketing_shock_risk/100) / forecast_base['NHITS'].mean()
    prob_prop_base_clients = model_prop.predict_proba(X_client_scaled)[:,1]
    prob_prop_stress_clients = np.clip(prob_prop_base_clients * stress_factor_risk, 0, 1)
    risks = ["Bajo" if p > 0.7 else "Medio" if p > 0.3 else "Alto" for p in prob_prop_stress_clients]

    risk_summary = pd.DataFrame({
        'Nivel de Riesgo (no compra)': ['Bajo', 'Medio', 'Alto'],
        'Número de Clientes': [risks.count('Bajo'), risks.count('Medio'), risks.count('Alto')],
        'Porcentaje (%)': [risks.count('Bajo')/len(risks)*100, risks.count('Medio')/len(risks)*100, risks.count('Alto')/len(risks)*100],
        'Propensión Promedio': [
            prob_prop_stress_clients[np.array(risks) == 'Bajo'].mean(),
            prob_prop_stress_clients[np.array(risks) == 'Medio'].mean(),
            prob_prop_stress_clients[np.array(risks) == 'Alto'].mean()
        ]
    }).round(2)

    st.write("Resumen de Riesgo Clientes bajo Stress (estilo Power BI)")
    st.dataframe(risk_summary.style.background_gradient(cmap='Greens', subset=['Porcentaje (%)']))

    st.write("Distribución Propensiones bajo Stress")
    fig, ax = plt.subplots()
    sns.histplot(prob_prop_stress_clients, bins=50, kde=True, ax=ax)
    st.pyplot(fig)

elif section == "Análisis de Sensibilidad Robusta":
    st.header("📊 Análisis de Sensibilidad Robusta")
    confidence_levels = st.slider("Rango Shock Confianza (%)", -40.0, 30.0, (-20.0, 10.0))
    marketing_levels = st.slider("Rango Shock Marketing (%)", -60.0, 60.0, (-30.0, 30.0))

    levels = np.linspace(confidence_levels[0], confidence_levels[1], 11)
    impact_sales = []
    base_sales = forecast_base['NHITS'].mean()

    for level in levels:
        factor = (1 + level/100) * (1 + marketing_levels[0]/100)  # Combinado
        stressed_sales = base_sales * factor
        impact_sales.append(stressed_sales)

    fig, ax = plt.subplots()
    ax.plot(levels, impact_sales, label=f'Marketing {marketing_levels[0]}% a {marketing_levels[1]}%')
    ax.set_xlabel('Shock Confianza Consumidor (%)')
    ax.set_ylabel('Ventas Pólizas Esperadas')
    ax.legend()
    st.pyplot(fig)

elif section == "Simulación de Monte Carlo":
    st.header("🎲 Simulación de Monte Carlo")
    n_mc_sim = st.slider("Número de Simulaciones", 500, 10000, 2000, step=500)
    confidence_shock_mc = st.slider("Shock Confianza MC (%)", -30.0, 20.0, -20.0, key="mc_conf")
    marketing_shock_mc = st.slider("Shock Marketing MC (%)", -50.0, 50.0, -30.0, key="mc_mark")

    mc_sales = []
    mc_revenue = []
    average_premium = df_aggregate['average_premium'].mean()

    forecast_stress_mc = forecast_base.copy()
    forecast_stress_mc['NHITS'] *= (1 + confidence_shock_mc/100) * (1 + marketing_shock_mc/100)

    for _ in range(n_mc_sim):
        noise = np.random.normal(1, 0.15, size=len(forecast_stress_mc))
        stressed_sales = forecast_stress_mc['NHITS'] * noise
        total_sales = stressed_sales.mean()
        revenue = total_sales * average_premium * 12
        mc_sales.append(total_sales)
        mc_revenue.append(revenue)

    mc_sales = np.array(mc_sales)
    mc_revenue = np.array(mc_revenue)

    mc_summary = pd.DataFrame({
        'Métrica': ['Ventas Pólizas Mensuales', 'Ingresos Anuales Seguros ($)'],
        'Media': [mc_sales.mean(), mc_revenue.mean()],
        'Mediana': [np.median(mc_sales), np.median(mc_revenue)],
        'VaR 90%': [np.percentile(mc_sales, 90), np.percentile(mc_revenue, 90)],
        'VaR 95%': [np.percentile(mc_sales, 95), np.percentile(mc_revenue, 95)],
        'VaR 99%': [np.percentile(mc_sales, 99), np.percentile(mc_revenue, 99)],
        'Máximo': [mc_sales.max(), mc_revenue.max()]
    }).round(0)

    st.write("Resumen Simulación Monte Carlo ")


    st.dataframe(
    mc_summary.style
    .format({col: "{:,.0f}" for col in mc_summary.select_dtypes(include=['float','int']).columns})
    .background_gradient(cmap='Reds', subset=['VaR 99%'])
    )

    col1, col2 = st.columns(2)
    with col1:
        st.write("Distribución Ventas Pólizas")
        fig, ax = plt.subplots()
        sns.histplot(mc_sales, kde=True, ax=ax)
        st.pyplot(fig)
    with col2:
        st.write("Distribución Ingresos Seguros")
        fig, ax = plt.subplots()
        sns.histplot(mc_revenue, kde=True, ax=ax)
        st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.markdown("**Creado por @NarenCastellon** | Especialización en Forecasting & IA 2026")