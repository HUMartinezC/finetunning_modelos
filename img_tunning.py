from datasets import load_dataset, DatasetDict
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np

dataset = load_dataset("cifar10")

train_dataset = dataset['train'].shuffle(seed=21).select(range(500))
test_dataset = dataset['test'].shuffle(seed=21).select(range(100))

split = train_dataset.train_test_split(test_size=0.1, seed=42)
dataset_final = DatasetDict({
    'train': split['train'],
    'validation': split['test'],
    'test': test_dataset
})

model_name = "google/vit-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_name)

def preprocess(example):
    processed = processor(images=example["img"], return_tensors=None)
    example["pixel_values"] = processed["pixel_values"][0]  # shape (3,224,224)
    return example

dataset_final = dataset_final.map(preprocess, batched=False)

dataset_final = dataset_final.remove_columns(['img'])

num_labels = 10
model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

metric = evaluate.load("accuracy")
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return metric.compute(predictions=preds, references=p.label_ids)

training_args = TrainingArguments(
    output_dir="./vit-cifar10",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,  # muy pequeño para CPU
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    num_train_epochs=1,  # prueba rápida
    logging_dir="./logs",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_final['train'],
    eval_dataset=dataset_final['validation'],
    compute_metrics=compute_metrics,
)

trainer.train()

test_results = trainer.evaluate(dataset_final['test'])
print("Test Accuracy:", test_results['eval_accuracy'])

model.save_pretrained("./mi_modelo_vit_cifar10")
processor.save_pretrained("./mi_modelo_vit_cifar10")