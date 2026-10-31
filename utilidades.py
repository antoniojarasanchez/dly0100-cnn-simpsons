# Utilidades de datos, visualización y exportación del Parcial 2.
# Las rutas de archivos y carpetas se reciben como objetos Path.
import json
import re
import tarfile
from collections import Counter

import cv2
import gdown
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

EXTENSIONES = {".jpg", ".jpeg", ".png"}


def mostrar_tabla(titulo, tabla):
    # tabla es un DataFrame; el formato cambia solo la presentación.
    display(Markdown(f"**{titulo}**"))
    formato = tabla.style.format(precision=4, decimal=",", thousands=".")
    if "support" in tabla.columns:
        formato = formato.format(
            "{:,.0f}", subset=["support"], decimal=",", thousands="."
        )
    display(formato)


def preparar_datos(url, archivo, carpeta, dataset_dir):
    # La marca se escribe al terminar la extracción y evita repetirla.
    marcador = carpeta / ".extraction_complete"
    if marcador.is_file():
        return
    if not archivo.is_file():
        descargado = gdown.download(url, str(archivo), quiet=False, fuzzy=True)
        if descargado is None or not archivo.is_file():
            # Evita reutilizar una descarga incompleta en la próxima ejecución.
            archivo.unlink(missing_ok=True)
            raise RuntimeError(
                f"No se pudo descargar {archivo.name}. "
                "Revisa la conexión y el acceso al archivo en Google Drive."
            )
    with tarfile.open(archivo, "r:gz") as comprimido:
        comprimido.extractall(path=dataset_dir, filter="data")
    if not carpeta.is_dir():
        raise FileNotFoundError(
            f"No se encontró {carpeta} después de extraer los datos."
        )
    marcador.write_text("complete\n", encoding="utf-8")


def cargar_datos(carpeta, id_por_clase, img_size, es_prueba=False):
    # id_por_clase asigna etiquetas consecutivas desde 0.
    # Devuelve imágenes RGB uint8 (n, img_size, img_size, 3), sin normalizar,
    # además de etiquetas, rutas relativas y conteos de las clases excluidas.
    imagenes, etiquetas, archivos = [], [], []
    excluidos = Counter()
    print(f"Buscando imágenes en {carpeta}...", flush=True)
    rutas = sorted(
        p for p in carpeta.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONES
    )
    print(f"Archivos encontrados: {len(rutas)}", flush=True)
    siguiente_avance = 10
    for indice, ruta in enumerate(rutas, start=1):
        if siguiente_avance < 100 and indice - 1 >= len(rutas) * siguiente_avance / 100:
            print(
                f"Revisados: {siguiente_avance}% ({indice - 1}/{len(rutas)})",
                flush=True,
            )
            siguiente_avance += 10
        if es_prueba:
            # El sufijo numérico identifica la imagen; el prefijo identifica al personaje.
            nombre = re.fullmatch(r"(.+)_(\d+)", ruta.stem)
            if nombre is None or ruta.parent != carpeta:
                raise ValueError(
                    f"Nombre o ubicación de prueba inesperados: {ruta.name}"
                )
            clase = nombre.group(1)
        else:
            if ruta.parent.parent != carpeta:
                raise ValueError(f"Ubicación de entrenamiento inesperada: {ruta.name}")
            clase = ruta.parent.name
        if clase not in id_por_clase:
            excluidos[clase] += 1
            continue
        imagen = cv2.imdecode(
            np.frombuffer(ruta.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if imagen is None:
            raise ValueError(f"No se pudo leer {ruta}")
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        imagenes.append(cv2.resize(imagen, (img_size, img_size)))
        etiquetas.append(id_por_clase[clase])
        archivos.append(ruta.relative_to(carpeta).as_posix())
    print(f"Revisados: 100% ({len(rutas)}/{len(rutas)})", flush=True)
    if set(etiquetas) != set(range(len(id_por_clase))):
        raise ValueError("Faltan clases seleccionadas en los datos cargados.")
    return np.stack(imagenes), np.array(etiquetas), archivos, dict(excluidos)


def guardar_figura(figura, carpeta, nombre):
    carpeta.mkdir(parents=True, exist_ok=True)
    figura.savefig(carpeta / nombre, dpi=180, bbox_inches="tight")


def mostrar_distribucion(y, imagenes, nombres, carpeta_figuras):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(nombres, np.bincount(y, minlength=len(nombres)))
    ax.set(
        ylabel="Imágenes",
        title="Distribución antes de dividir entrenamiento y validación",
    )
    ax.tick_params(axis="x", rotation=75)
    plt.tight_layout()
    guardar_figura(fig, carpeta_figuras, "distribucion.png")
    plt.show()
    plt.close(fig)
    fig, axes = plt.subplots(3, 6, figsize=(14, 7))
    for label, ax in enumerate(axes.ravel()):
        ax.imshow(imagenes[np.flatnonzero(y == label)[0]])
        ax.set_title(nombres[label], fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    guardar_figura(fig, carpeta_figuras, "muestras_clases.png")
    plt.show()
    plt.close(fig)


def graficar_comparacion(resumen, runs, configuraciones, carpeta_figuras):
    # resumen es un DataFrame; runs conserva los historiales de las seis configuraciones.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        resumen["configuracion"],
        resumen["val_accuracy_media"],
        yerr=resumen["val_accuracy_std"],
        capsize=5,
    )
    ax.set(
        xlabel="Configuración",
        ylabel="Accuracy de validación",
        title="Media y desviación entre semillas",
    )
    plt.tight_layout()
    guardar_figura(fig, carpeta_figuras, "comparacion_validacion.png")
    plt.show()
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True, sharey=True)
    for nombre, ax in zip(configuraciones, axes.ravel()):
        for run in runs:
            if run["config_id"] == nombre:
                (linea,) = ax.plot(
                    np.arange(1, run["epocas"] + 1),
                    run["historial"]["val_accuracy"],
                    label=str(run["semilla"]),
                )
                epoca = run["mejor_epoca"]
                ax.scatter(
                    epoca,
                    run["historial"]["val_accuracy"][epoca - 1],
                    marker="X",
                    s=55,
                    color=linea.get_color(),
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                )
        ax.set(title=nombre, xlabel="Época", ylabel="Accuracy de validación")
        ax.legend(title="Semilla", fontsize=8)
    fig.suptitle(
        "Accuracy de validación por RUN - X: checkpoint de mínima pérdida", fontsize=13
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    guardar_figura(fig, carpeta_figuras, "curvas_validacion.png")
    plt.show()
    plt.close(fig)


def graficar_historial(historial_principal, mejor_epoca, carpeta_figuras):
    # El índice del DataFrame corresponde a las épocas, comenzando en 1.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for metrica, ax in zip(("loss", "accuracy"), axes):
        ax.plot(
            historial_principal.index,
            historial_principal[metrica],
            label="Entrenamiento durante fit",
        )
        ax.plot(
            historial_principal.index,
            historial_principal["val_" + metrica],
            label="Validación",
        )
        ax.axvline(mejor_epoca, color="gray", linestyle="--", label="Checkpoint")
        ax.set(xlabel="Época", ylabel=metrica)
        ax.legend()
    plt.tight_layout()
    guardar_figura(fig, carpeta_figuras, "curvas_principal.png")
    plt.show()
    plt.close(fig)


def graficar_matriz(cm, nombres, carpeta_figuras):
    # Filas: clases reales. Columnas: predicciones. Cada clase debe tener ejemplos.
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    imagen = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set(
        xticks=np.arange(len(nombres)),
        yticks=np.arange(len(nombres)),
        xticklabels=nombres,
        yticklabels=nombres,
        xlabel="Predicción",
        ylabel="Personaje real",
        title="Matriz de confusión normalizada - modelo principal",
    )
    plt.setp(ax.get_xticklabels(), rotation=90)
    for i in range(len(nombres)):
        for j in range(len(nombres)):
            if cm[i, j]:
                ax.text(
                    j,
                    i,
                    f"{cm_norm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                )
    fig.colorbar(imagen, ax=ax)
    plt.tight_layout()
    guardar_figura(fig, carpeta_figuras, "matriz_confusion.png")
    plt.show()
    plt.close(fig)


def mostrar_ejemplos(
    imagenes, etiquetas, probabilidades, nombres, carpeta_figuras, semilla=2026
):
    # probabilidades tiene una fila por imagen y una columna por clase.
    predicciones = probabilidades.argmax(axis=1)
    ejemplos = {}
    for titulo, mascara in [
        ("Aciertos", predicciones == etiquetas),
        ("Errores", predicciones != etiquetas),
    ]:
        candidatos = np.flatnonzero(mascara)
        elegidos = np.random.default_rng(semilla).choice(
            candidatos, size=min(6, len(candidatos)), replace=False
        )
        ejemplos[titulo] = elegidos.tolist()
        if not len(elegidos):
            (carpeta_figuras / (titulo.lower() + ".png")).unlink(missing_ok=True)
            print(titulo + ": no hay ejemplos en esta ejecución.")
            continue
        fig, axes = plt.subplots(2, 3, figsize=(10, 9.5))
        fig.subplots_adjust(
            left=0.02, right=0.98, top=0.89, bottom=0.02, hspace=0.55, wspace=0.12
        )
        for ax in axes.ravel():
            ax.axis("off")
        for ax, indice in zip(axes.ravel(), elegidos):
            ax.imshow(imagenes[indice])
            ax.set_title(
                f"Real: {nombres[etiquetas[indice]]}\nPredicción: {nombres[predicciones[indice]]}\n"
                f"Confianza: {probabilidades[indice].max():.1%}",
                fontsize=12,
                pad=10,
            )
        fig.suptitle(titulo + " - modelo principal", fontsize=15, y=0.98)
        guardar_figura(fig, carpeta_figuras, titulo.lower() + ".png")
        plt.show()
        plt.close(fig)
    return ejemplos


def guardar_respaldo(contenido, ruta):
    # El registro completo queda fuera del notebook para mantenerlo liviano.
    ruta.write_text(
        json.dumps(contenido, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
