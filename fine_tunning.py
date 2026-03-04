# -------------------------------
# Fine-tuning DistilBERT en clasificación de noticias
# -------------------------------

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import evaluate
import numpy as np

dataset = load_dataset("ag_news")

# Selección de un subset pequeño para prueba rápida
train_dataset = dataset["train"].shuffle(seed=21).select(range(500))
test_dataset = dataset["test"].shuffle(seed=21).select(range(100))

# Separar validación del train
split = train_dataset.train_test_split(test_size=0.1, seed=42)
dataset_final = DatasetDict({
    'train': split['train'],
    'validation': split['test'],
    'test': test_dataset
})

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train = dataset_final['train'].map(tokenize_function, batched=True)
tokenized_val = dataset_final['validation'].map(tokenize_function, batched=True)
tokenized_test = dataset_final['test'].map(tokenize_function, batched=True)

num_labels = 4  # ag_news tiene 4 clases
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./distilbert-ag-news",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    num_train_epochs=1,
    logging_dir="./logs",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

trainer.train()

test_results = trainer.evaluate(tokenized_test)
print("Test Accuracy:", test_results['eval_accuracy'])

model.save_pretrained("./mi_modelo_entrenado")
tokenizer.save_pretrained("./mi_modelo_entrenado")
