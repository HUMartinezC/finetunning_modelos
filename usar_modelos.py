from transformers import pipeline
from PIL import Image


AG_NEWS_LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

CIFAR10_LABELS = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def print_header(title):
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")


def traducir_label(label, mapping):
    if not label.startswith("LABEL_"):
        return label
    try:
        indice = int(label.split("_")[1])
        return mapping.get(indice, label)
    except (IndexError, ValueError):
        return label


def mostrar_top_k(resultados, top_k=3, label_mapping=None):
    label_mapping = label_mapping or {}
    for i, pred in enumerate(resultados[:top_k], start=1):
        etiqueta = pred.get("label", "N/A")
        etiqueta_legible = traducir_label(etiqueta, label_mapping)
        score = pred.get("score", 0.0)
        print(f"{i}. {etiqueta_legible:<20} | score: {score:.4f}")


def inferencia_texto():
    clasificador_texto = pipeline(
        "text-classification",
        model="./mi_modelo_entrenado",
        tokenizer="./mi_modelo_entrenado",
    )

    frases = [
        "The stock market crashed today due to economic uncertainty.",
        "The local team won the championship after a dramatic final.",
    ]

    print_header("Modelo de texto (DistilBERT fine-tuned en AG News)")
    for frase in frases:
        print(f"\nTexto: {frase}")
        resultados = clasificador_texto(frase, top_k=4)
        mostrar_top_k(resultados, top_k=4, label_mapping=AG_NEWS_LABELS)


def inferencia_imagen():
    clasificador_img = pipeline(
        "image-classification",
        model="./mi_modelo_vit_cifar10",
        image_processor="./mi_modelo_vit_cifar10",
    )

    imagenes = {
        "truck.jpeg": Image.open("truck.jpeg"),
        "bird.jpg": Image.open("bird.jpg"),
    }

    print_header("Modelo de imagen (ViT fine-tuned en CIFAR-10)")
    for nombre, imagen in imagenes.items():
        print(f"\nImagen: {nombre}")
        resultados = clasificador_img(imagen, top_k=5)
        mostrar_top_k(resultados, top_k=5, label_mapping=CIFAR10_LABELS)


if __name__ == "__main__":
    inferencia_texto()
    inferencia_imagen()