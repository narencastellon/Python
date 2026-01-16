import streamlit as st
from transformers import pipeline
import torch

# 1. OPTIMIZACIÓN PARA APPLE M1 (MPS)
device = 0 if torch.backends.mps.is_available() else -1

@st.cache_resource
def cargar_cerebro_ia():
    return pipeline("zero-shot-classification", 
                    model="facebook/bart-large-mnli", 
                    device=device)

clasificador = cargar_cerebro_ia()

# 2. DEFINICIÓN DE INTENCIONES BANCARIAS
intenciones = [
    "consultar saldo",
    "bloquear tarjeta de crédito",
    "solicitar préstamo personal",
    "transferencia internacional",
    "hablar con asesor humano",
    "hola",
    "si quiero la simulacion",
    "cual es tu nombre",
]

# 3. MAPEO DE RESPUESTAS
respuestas = {
    "consultar saldo": "💰 Tu saldo actual en la cuenta terminada en *4590 es de **$2,450.00 USD**.",
    "bloquear tarjeta de crédito": "🚨 He iniciado el protocolo de bloqueo. ¿Fue por robo o extravío? Por favor, confirma para proceder.",
    "solicitar préstamo personal": "📈 Tenemos una oferta pre-aprobada para ti con una tasa del 12% anual. ¿Te gustaría ver la simulación?",
    "transferencia internacional": "🌍 Para transferencias al exterior, necesito el código SWIFT/BIC del banco destino. ¿Lo tienes a mano?",
    "hablar con asesor humano": "👨‍💼 Entendido. Te estoy transfiriendo con un ejecutivo. El tiempo de espera es de 2 minutos.",
    "hola": "👋 ¡Hola! Bienvenido a tu Banca Digital. Soy tu asistente virtual inteligente. ¿En qué puedo apoyarte hoy?",
    "desconocido": "🤔 No estoy seguro de haber entendido. ¿Podrías reformular tu consulta? (Ej: 'Quiero bloquear mi tarjeta')",
    "cual es tu nombre": "cual es tu nombre",
    "si quiero la simulacion": "Por ejemplo, si pides 10,000 a un año al 12%, pagarías 12 cuotas de $888.49, y al final habrás pagado $661.88 de intereses."
}

# 4. INTERFAZ DE USUARIO (Streamlit)
st.set_page_config(page_title="IA Banking Bot", page_icon="🏦")
st.title("🏦 Asistente Bancario Inteligente")

# --- BOTÓN DE LIMPIEZA EN LA BARRA LATERAL ---
with st.sidebar:
    st.header("Configuración")
    # Al hacer clic en el botón, reiniciamos la lista de mensajes
    if st.button("🗑️ Limpiar historial de chat"):
        st.session_state.mensajes = []
        st.rerun() # Esto refresca la app para mostrar el chat vacío de inmediato

st.info("Este bot usa **Machine Learning** para entender tus intenciones sin necesidad de comandos fijos.")

# Historial de chat
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Mostrar mensajes anteriores
for m in st.session_state.mensajes:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 5. PROCESAMIENTO DE LA ENTRADA DEL USUARIO
if prompt := st.chat_input("Escribe tu consulta aquí..."):
    # Guardar y mostrar mensaje del usuario
    st.session_state.mensajes.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Analizando tu solicitud..."):
        # EL CORAZÓN DEL ML
        resultado = clasificador(prompt, candidate_labels=intenciones)
        
        intencion_detectada = resultado['labels'][0]
        confianza = resultado['scores'][0]

        if confianza > 0.40:
            respuesta_bot = respuestas[intencion_detectada]
        else:
            respuesta_bot = respuestas["desconocido"]

    # Mostrar respuesta del bot
    with st.chat_message("assistant"):
        st.markdown(respuesta_bot)
        st.caption(f"IA detectó: **{intencion_detectada}** (Confianza: {confianza:.2f})")

    # Guardar respuesta del bot en el historial
    st.session_state.mensajes.append({"role": "assistant", "content": respuesta_bot})