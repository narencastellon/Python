import pandas as pd  # Para manipulación de datos, creación de DataFrames y manejo de grandes datasets
import numpy as np   # Para generación de datos sintéticos numéricos, fechas aleatorias y operaciones matemáticas

import matplotlib.pyplot as plt  # Para visualizaciones básicas como gráficos de series temporales y forecasts
import seaborn as sns  # Para gráficos avanzados en EDA, como heatmaps o distribuciones

from datetime import datetime, timedelta  # Para generar timestamps realistas en consultas
import random  # Para selección aleatoria de categorías e intenciones

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split  # Para dividir el dataset en train/test para fine-tuning
from sklearn.metrics import accuracy_score, classification_report  # Para evaluar el modelo de clasificación de intenciones
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments  # Hugging Face para tokenizer, modelo pre-entrenado, fine-tuning y entrenamiento
from transformers import TextClassificationPipeline  # Para pipeline de inferencia simple en el chatbot
from datasets import Dataset  # Para convertir DataFrame a formato Hugging Face Dataset para fine-tuning
import torch  # Backend para Hugging Face (verifica GPU si disponible)
import warnings
warnings.filterwarnings('ignore')  # Ignorar warnings menores para salida limpia

chat_df = pd.read_csv("bank_chat_queries.csv")

# Paso 2: Análisis Exploratorio de Datos (EDA)
# Resumen estadístico para entender distribuciones
print("\nEstadísticas descriptivas:")
print(chat_df.describe())

# Distribución de intenciones
plt.figure(figsize=(10, 6))
sns.countplot(y='intention', data=chat_df, palette='viridis')
plt.title('Distribución de Intenciones en Consultas')
plt.xlabel('Conteo')
plt.show()


# Longitud de queries por intención (para ver variabilidad)
chat_df['query_length'] = chat_df['query_text'].apply(len)
plt.figure(figsize=(12, 6))
sns.boxplot(x='intention', y='query_length', data=chat_df)
plt.title('Longitud de Consultas por Intención')
plt.xticks(rotation=45)
plt.show()

# Paso 3: Preprocesamiento para Fine-Tuning con Hugging Face
# Dividir en train/test (80/20)
train_df, test_df = train_test_split(chat_df, test_size=0.2, stratify=chat_df['intention'], random_state=42)

# Codificar labels (intenciones a números para clasificación)
label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['intention'])
test_df['label'] = label_encoder.transform(test_df['intention'])

# Convertir a Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df[['query_text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['query_text', 'label']])

# Cargar tokenizer y modelo base (distilbert para eficiencia, fine-tune para clasificación)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

# Tokenizar datasets
def tokenize_function(examples):
    return tokenizer(examples['query_text'], padding='max_length', truncation=True, max_length=64)

train_tokenized = train_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)


# Set format for PyTorch
train_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

from transformers import EarlyStoppingCallback  # Agregar para load_best_model_at_end

# Paso 4: Fine-Tuning del Modelo con Trainer
# Configurar argumentos de entrenamiento (epochs bajos para demo, ajusta para precisión)
training_args = TrainingArguments(
    output_dir='./bank_intent_model',
    num_train_epochs=3,  # 3 epochs para fine-tuning rápido
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    eval_strategy='epoch',  # Cambiado de 'evaluation_strategy' (deprecated en versiones nuevas)
    save_strategy='epoch',  # Añadido para coincidir con eval_strategy y evitar error con load_best_model_at_end
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy'
)

# Función para métricas
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc}

# Trainer para fine-tuning (agregar callback para early stopping si load_best_model_at_end=True)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Agregar para load_best_model_at_end
)

# Entrenar modelo
print("Iniciando fine-tuning del modelo...")
trainer.train()

# Guardar modelo fine-tuned
trainer.save_model('./bank_intent_model_finetuned')
tokenizer.save_pretrained('./bank_intent_model_finetuned')
print("Modelo fine-tuned guardado.")

# Paso 5: Evaluación del Modelo
# Predicciones en test
predictions = trainer.predict(test_tokenized)
preds = np.argmax(predictions.predictions, axis=-1)
print("\nReporte de Clasificación:")
print(classification_report(test_df['label'], preds, target_names=label_encoder.classes_))

# Paso 6: Implementar el Chatbot Simple con el Modelo Fine-Tuned
# Usamos pipeline para inferencia rápida
classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Respuestas pre-definidas por intención (simulado chatbot)
responses = {
    'check_balance': "Su saldo actual es $1,234.56. ¿Necesita más detalles?",
    'report_fraud': "Hemos registrado su reporte de fraude. Un agente se contactará pronto.",
    'transfer_money': "Transferencia realizada exitosamente. Confirme el monto y destinatario.",
    'open_account': "Para abrir una cuenta, visite nuestra sucursal o app. ¿Qué tipo de cuenta desea?",
    'loan_inquiry': "Nuestras tasas de préstamo empiezan en 5%. ¿Cuánto necesita?",
    'card_activation': "Su tarjeta ha sido activada. Use PIN 1234 para primera transacción.",
    'payment_issue': "El pago falló por fondos insuficientes. Intente de nuevo o contacte soporte."
}

# Función para chatbot (loop simple en consola para demo; en producción, integra con web/app)
def run_chatbot():
    print("\n--- Chatbot Bancario ---")
    print("Escriba su consulta (o 'salir' para terminar):")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'salir':
            break
        result = classifier(user_input)[0]
        intention = result['label'].split('LABEL_')[1]  # Extraer label (0-6) y map a intención
        intention_label = label_encoder.inverse_transform([int(intention)])[0]
        response = responses.get(intention_label, "Lo siento, no entendí su consulta. ¿Puede reformular?")
        print(f"Bot: {response}")

# Ejecutar chatbot demo
if __name__ == "__main__":
    run_chatbot()

print("\n¡Ejemplo end-to-end completado! El chatbot usa ML para detectar intenciones y responder acorde.")
print("- Dataset: 100k queries sintéticas con variabilidad para fine-tuning realista.")
print("- Modelo: Fine-tuned DistilBERT para clasificación de intenciones (eficiente y preciso).")
print("- Chatbot: Loop simple; en producción, integra con Rasa para conversaciones multi-turn o Flask para web.")
print("- Recomendación: Para escalabilidad, usa Hugging Face Hub para hostear modelo; agrega RAG para respuestas dinámicas con datos usuario.")


