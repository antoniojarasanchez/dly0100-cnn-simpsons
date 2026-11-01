# Utilidades de datos, visualización y exportación del Parcial 2.
# Las rutas de archivos y carpetas se reciben como objetos Path.
import contextlib
import io
import json
import re
import shutil
import tarfile
import time
from collections import Counter

import cv2
import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
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


def empaquetar_resultados(carpeta):
    # El ZIP queda junto a la carpeta para no incluirse a sí mismo.
    return shutil.make_archive(
        str(carpeta), "zip", root_dir=carpeta.parent, base_dir=carpeta.name
    )


def crear_modelo(config, img_size, num_classes):
    # La configuración define arquitectura, dropout y tasa de aprendizaje.
    red = keras.Sequential(name="cnn_" + config["id"])
    red.add(keras.Input(shape=(img_size, img_size, 3)))
    if config["arquitectura"] == "simple":
        red.add(Conv2D(64, (3, 3), strides=1, padding="valid", activation="relu"))
        red.add(MaxPooling2D((2, 2)))
        if config["dropout"]:
            red.add(Dropout(config["dropout"]))
        unidades = 128
    elif config["arquitectura"] == "profunda":
        red.add(Conv2D(32, (3, 3), strides=1, padding="valid", activation="relu"))
        red.add(Conv2D(32, (3, 3), strides=1, padding="valid", activation="relu"))
        red.add(MaxPooling2D((2, 2)))
        if config["dropout"]:
            red.add(Dropout(config["dropout"]))
        red.add(Conv2D(64, (3, 3), strides=1, padding="valid", activation="relu"))
        red.add(MaxPooling2D((2, 2)))
        unidades = 512
    else:
        raise ValueError("Arquitectura desconocida.")
    red.add(Flatten())
    red.add(Dense(unidades, activation="relu"))
    if config["dropout"]:
        red.add(Dropout(config["dropout"]))
    red.add(Dense(num_classes, activation="softmax"))
    red.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["lr"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return red


def ejecutar_run(config, semilla, datos, ajustes, numero_run):
    # Los arreglos se reciben por referencia; no se copian los conjuntos.
    X_train, Y_train = datos["X_train"], datos["Y_train"]
    X_val, Y_val, y_val = datos["X_val"], datos["Y_val"], datos["y_val"]
    PESOS_DIR = ajustes["carpeta_pesos"]
    BATCH_SIZE = ajustes["batch_size"]
    MAX_EPOCHS = ajustes["max_epochs"]
    PACIENCIA = ajustes["paciencia"]
    run_id = f"{config['id']}_s{semilla}"
    pesos = PESOS_DIR / f"{run_id}.weights.h5"
    # Cada RUN empieza sin un checkpoint anterior.
    pesos.unlink(missing_ok=True)
    # Preparar un modelo nuevo para esta semilla.
    keras.backend.clear_session()
    keras.utils.set_random_seed(semilla)
    red = crear_modelo(config, ajustes["img_size"], ajustes["num_classes"])
    parada = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=PACIENCIA,
        restore_best_weights=True,
    )
    checkpoint = ModelCheckpoint(
        str(pesos),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )
    traza = io.StringIO()
    print(
        f"Entrenamiento {numero_run}/18 | {run_id} | "
        f"arquitectura={config['arquitectura']}, dropout={config['dropout']}, lr={config['lr']}",
        flush=True,
    )
    # Entrenar y conservar la traza completa.
    inicio = time.perf_counter()
    with contextlib.redirect_stdout(traza):
        entrenamiento = red.fit(
            X_train,
            Y_train,
            validation_data=(X_val, Y_val),
            batch_size=BATCH_SIZE,
            epochs=MAX_EPOCHS,
            shuffle=True,
            verbose=2,
            callbacks=[parada, checkpoint],
        )
    segundos = time.perf_counter() - inicio
    historial = {
        k: [float(v) for v in valores] for k, valores in entrenamiento.history.items()
    }
    assert all(np.isfinite(valores).all() for valores in historial.values())
    # Se evalúan los pesos del mínimo val_loss, no los de la última época.
    red.load_weights(pesos)
    metricas = red.evaluate(
        X_val, Y_val, batch_size=BATCH_SIZE, verbose=0, return_dict=True
    )
    predicciones = red.predict(X_val, batch_size=BATCH_SIZE, verbose=0).argmax(axis=1)
    mejor_epoca = int(np.argmin(historial["val_loss"])) + 1
    assert np.isclose(metricas["loss"], min(historial["val_loss"]), atol=1e-5)
    assert np.isclose(
        metricas["accuracy"], accuracy_score(y_val, predicciones), atol=1e-6
    )
    # Registrar las métricas, las predicciones y el historial.
    resultado = {
        "id": run_id,
        "config_id": config["id"],
        "semilla": semilla,
        "parametros": int(red.count_params()),
        "epocas": len(historial["loss"]),
        "mejor_epoca": mejor_epoca,
        "segundos": segundos,
        "detencion": "early_stopping" if parada.stopped_epoch > 0 else "max_epochs",
        "val_loss": float(metricas["loss"]),
        "val_accuracy": float(metricas["accuracy"]),
        "val_f1_macro": float(
            f1_score(y_val, predicciones, average="macro", zero_division=0)
        ),
        "predicciones_val": predicciones.tolist(),
        "historial": historial,
        "traza_fit": re.sub(r"\x1b\[[0-9;]*m", "", traza.getvalue()),
        "checkpoint": pesos.as_posix(),
    }
    print(
        f"{run_id}: val_accuracy={metricas['accuracy']:.4f}; mejor época={mejor_epoca}",
        flush=True,
    )
    del red
    return resultado


def resumir_runs(resultados, ids, semillas):
    # Cada configuración debe tener un resultado por cada semilla indicada.
    filas = []
    for nombre in ids:
        runs = [run for run in resultados if run["config_id"] == nombre]
        assert sorted(run["semilla"] for run in runs) == sorted(semillas)
        filas.append(
            {
                "configuracion": nombre,
                "parametros": runs[0]["parametros"],
                "val_accuracy_media": float(np.mean([r["val_accuracy"] for r in runs])),
                "val_accuracy_std": float(
                    np.std([r["val_accuracy"] for r in runs], ddof=1)
                ),
                "val_f1_macro_media": float(np.mean([r["val_f1_macro"] for r in runs])),
                "val_loss_media": float(np.mean([r["val_loss"] for r in runs])),
            }
        )
    return pd.DataFrame(filas)


def elegir_configuracion(resultados, ids, semillas):
    # El orden fija el criterio principal y los desempates de la comparación.
    tabla = resumir_runs(resultados, ids, semillas).sort_values(
        ["val_accuracy_media", "val_f1_macro_media", "parametros", "configuracion"],
        ascending=[False, False, True, True],
    )
    return str(tabla.iloc[0]["configuracion"])


def entrenar_configuraciones(
    ids, configuraciones, semillas, datos, ajustes, resultados
):
    # Cada entrenamiento se agrega a la lista recibida, conservando su orden.
    for nombre in ids:
        for semilla in semillas:
            resultado = ejecutar_run(
                configuraciones[nombre],
                semilla,
                datos,
                ajustes,
                len(resultados) + 1,
            )
            resultados.append(resultado)


def evaluar_configuracion(
    config, resultados, semillas, semilla_principal, datos, ajustes, nombres
):
    # Prueba se evalúa después de elegir la configuración por validación.
    X_train, Y_train = datos["X_train"], datos["Y_train"]
    X_val, Y_val = datos["X_val"], datos["Y_val"]
    X_test, Y_test, y_test = datos["X_test"], datos["Y_test"], datos["y_test"]
    BATCH_SIZE = ajustes["batch_size"]
    num_classes = ajustes["num_classes"]
    evaluaciones = []
    prob_principal = None
    for semilla in semillas:
        run = next(
            r
            for r in resultados
            if r["config_id"] == config["id"] and r["semilla"] == semilla
        )
        keras.backend.clear_session()
        red = crear_modelo(config, ajustes["img_size"], ajustes["num_classes"])
        if semilla == semilla_principal:
            red.summary()
        red.optimizer.build(red.trainable_variables)
        red.load_weights(run["checkpoint"])
        probabilidades = red.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        assert np.isfinite(probabilidades).all() and np.allclose(
            probabilidades.sum(axis=1), 1, atol=1e-5
        )
        predicciones = probabilidades.argmax(axis=1)
        metricas = red.evaluate(
            X_test, Y_test, batch_size=BATCH_SIZE, verbose=0, return_dict=True
        )
        assert np.isclose(
            metricas["accuracy"], accuracy_score(y_test, predicciones), atol=1e-6
        )
        reporte = classification_report(
            y_test,
            predicciones,
            labels=list(range(num_classes)),
            target_names=nombres,
            output_dict=True,
            zero_division=0,
        )
        evaluaciones.append(
            {
                "semilla": semilla,
                "loss": float(metricas["loss"]),
                "accuracy": float(metricas["accuracy"]),
                "precision_macro": reporte["macro avg"]["precision"],
                "recall_macro": reporte["macro avg"]["recall"],
                "f1_macro": reporte["macro avg"]["f1-score"],
                "f1_weighted": reporte["weighted avg"]["f1-score"],
                "reporte": reporte,
                "matriz": confusion_matrix(
                    y_test, predicciones, labels=list(range(num_classes))
                ).tolist(),
                "predicciones": predicciones.tolist(),
            }
        )
        if semilla == semilla_principal:
            prob_principal = probabilidades.copy()
            metricas_train = red.evaluate(
                X_train, Y_train, batch_size=BATCH_SIZE, verbose=0, return_dict=True
            )
            metricas_val = red.evaluate(
                X_val, Y_val, batch_size=BATCH_SIZE, verbose=0, return_dict=True
            )
            brecha = {
                "train": {k: float(v) for k, v in metricas_train.items()},
                "validacion": {k: float(v) for k, v in metricas_val.items()},
                "train_menos_val_accuracy": float(
                    metricas_train["accuracy"] - metricas_val["accuracy"]
                ),
            }
        del red
    return evaluaciones, prob_principal, brecha
