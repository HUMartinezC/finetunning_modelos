from transformers import pipeline 
clasificador = pipeline("sentiment-analysis", model="./mi_modelo_entrenado", tokenizer="./mi_modelo_entrenado") 
frase = "This movie was surprisingly good, I enjoyed it a lot!" 
resultado = clasificador(frase) 
print(resultado)
