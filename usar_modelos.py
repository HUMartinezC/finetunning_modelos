from transformers import pipeline
from PIL import Image
import requests

clasificador_pre = pipeline(
    "text-classification",
    model="distilbert-base-uncased",
    tokenizer="distilbert-base-uncased"
)
frase = "The stock market crashed today due to economic uncertainty."

resultado_pre = clasificador_pre(frase)
# print("Resultado modelo original:", resultado_pre)

clasificador_img = pipeline(
    "image-classification",
    model="./mi_modelo_vit_cifar10",
    feature_extractor="./mi_modelo_vit_cifar10"  # se usa processor guardado
)

# Imagen de ejemplo (puedes usar local o URL)
truck = Image.open("truck.jpeg")
bird = Image.open("bird.jpg")

resultado_img = clasificador_img(imagen)
print("Resultado ViT fine-tuned:", resultado_img)