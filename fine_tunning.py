# from datasets import load_dataset

# dataset = load_dataset("<nombre_dataset_hugging_face>")

# train_dataset = dataset["train"].shuffle(seed=21).select(range(500))
# test_dataset = dataset["test"].shuffle(seed=21).select(range(100))

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("<modelo_huggingface>")

# def tokenize_function(examples):
#     		return tokenizer(examples["text"], padding="max_length", truncation=True)

# tokenized_train = train_dataset.map(tokenize_function, batched=True)
# tokenized_test = test_dataset.map(tokenize_function, batched=True)

# from transformers import AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained("<nombre_modelo_hugginface>", num_labels=<numero_clases_a_clasificar>)

# import evaluate
# metric = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)	

# from transformers import TrainingArguments
# training_args = TrainingArguments(
# output_dir="test_trainer", 	#Ruta donde se guardará nuestro entrenamiento
# evaluation_strategy="epoch",	#Define como realizar cada validación del entrenamiento
#                             #Por epoch, es cada vez que realiza una pasada completa por el dataset
# per_device_train_batch_size=2, # El número de elementos que se procesan a la vez en el entrenamiento
# per_device_eval_batch_size=2,  # El número de elementos que se procesan a la vez al evaluar
# learning_rate=2e-5, 	    # Ratio de parendizaje
# num_train_epochs=1,	    # Número de épocas
# )

# from transformers import Trainer

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train,
#     eval_dataset=tokenized_test,
#     compute_metrics=compute_metrics,
# )

# trainer.train()

# model.save_pretrained("./mi_modelo_entrenado") 

# tokenizer.save_pretrained("./mi_modelo_entrenado")


# -------------------------------
# Fine-tuning DistilBERT en clasificación de noticias (CPU friendly)
# -------------------------------

# 1️⃣ Instalar librerías necesarias
# !pip install transformers datasets evaluate

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import evaluate
import numpy as np

# 2️⃣ Cargar dataset (ejemplo: ag_news)
dataset = load_dataset("ag_news")

# 3️⃣ Selección de un subset pequeño para prueba rápida (CPU-friendly)
train_dataset = dataset["train"].shuffle(seed=21).select(range(500))
test_dataset = dataset["test"].shuffle(seed=21).select(range(100))

# 4️⃣ Separar validación del train
split = train_dataset.train_test_split(test_size=0.1, seed=42)
dataset_final = DatasetDict({
    'train': split['train'],
    'validation': split['test'],
    'test': test_dataset
})

# 5️⃣ Cargar tokenizer de DistilBERT
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 6️⃣ Tokenización segura con max_length
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train = dataset_final['train'].map(tokenize_function, batched=True)
tokenized_val = dataset_final['validation'].map(tokenize_function, batched=True)
tokenized_test = dataset_final['test'].map(tokenize_function, batched=True)

# 7️⃣ Cargar modelo DistilBERT para clasificación
num_labels = 4  # ag_news tiene 4 clases
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

# 8️⃣ Métrica
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 9️⃣ TrainingArguments (CPU-friendly)
training_args = TrainingArguments(
    output_dir="./distilbert-ag-news",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    num_train_epochs=1,  # probar primero 1 epoch
    logging_dir="./logs",
    load_best_model_at_end=True,
)

# 🔟 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# 1️⃣1️⃣ Entrenar
trainer.train()

# 1️⃣2️⃣ Evaluar en test final
test_results = trainer.evaluate(tokenized_test)
print("Test Accuracy:", test_results['eval_accuracy'])

# 1️⃣3️⃣ Guardar modelo y tokenizer
model.save_pretrained("./mi_modelo_entrenado")
tokenizer.save_pretrained("./mi_modelo_entrenado")
