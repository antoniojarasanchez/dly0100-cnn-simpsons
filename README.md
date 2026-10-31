# Clasificación de personajes de Los Simpson con CNN

Trabajo del Parcial 2 del curso **DLY0100 - Deep Learning**, desarrollado por Antonio Jara y Florencia Soto.

El objetivo es desarrollar y evaluar redes neuronales convolucionales con TensorFlow/Keras para clasificar imágenes de 18 personajes de Los Simpson, comparando arquitecturas, regularización y tasas de aprendizaje.

El dataset utilizado se atribuye a [Alexandre Attia](https://github.com/alexattia).

## Arquitecturas y comparación

Se adaptan dos arquitecturas del laboratorio `2.2.3_Redes_Convolucionales.ipynb`:

- **Simple:** Conv64, max pooling, Flatten, Dense128 y salida softmax de 18 clases.
- **Profunda:** Conv32, Conv32, max pooling, Conv64, max pooling, Flatten, Dense512 y salida softmax de 18 clases.

Se comparan seis configuraciones, con variantes de dropout y tasa de aprendizaje. A su vez, cada configuración se entrena con las semillas `42`, `123` y `456`: 18 entrenamientos en total. La configuración se elige comparando sus resultados en validación y el conjunto de prueba se evalúa después.

Las arquitecturas, el entrenamiento, la selección y el cálculo de métricas permanecen visibles en el notebook.

## Ejecución en Colab

El notebook fue desarrollado pensando en su ejecución en Google Colab, pero puede ejecutarse de manera local con los ajustes necesarios. En Colab, bastará con seleccionar **Ejecutar todo** para completar el proceso de inicio a fin de forma automática, sin necesidad de intervención manual.

La ejecución comienza descargando e importando `utilidades.py` desde este repositorio. Luego se prepara el dataset: los archivos se descargan desde Google Drive y se extraen automáticamente cuando es necesario. Las imágenes quedan en `data/simpsons/` para entrenamiento y `data/simpsons_testset/` para prueba; en Colab, la carpeta base suele ser `/content/data/`.

Con los datos preparados, se entrenan las seis configuraciones, se elige una mediante validación, se evalúa en prueba y se generan los resultados.

Por ahora, el dataset se aloja en Google Drive para efectos prácticos. Más adelante se deberá revisar ese alojamiento para asegurar su disponibilidad a largo plazo.

## Resultados

Al ejecutar el notebook, se crea `resultados/` con los siguientes archivos:

- **CSV:** métricas por entrenamiento, configuración, semilla y personaje, además de conteos y confusiones. `resumen_validacion.csv` incluye la pérdida media; `evaluacion_modelo_principal.csv` y `resumen_prueba.csv` reúnen las evaluaciones utilizadas por el informe.
- **`resultados.json`:** registro completo del experimento, con configuraciones, historiales, métricas y predicciones.
- **`figuras/`:** gráficos y muestras de imágenes en PNG, exportados al mostrarse en el notebook.

Al terminar, el notebook comprime la carpeta en `resultados.zip` e inicia su descarga en Colab. Basta con compartir ese ZIP para revisar los resultados y actualizar el informe. Para la entrega, guarda también el notebook con sus salidas.

Si alguien deseara consultar o analizar una ejecución ya guardada, puede descargar estos archivos desde este mismo repositorio y cargarlos en Colab, sin necesidad de repetir los entrenamientos ni ejecutar todo el notebook.

## Estructura del repositorio

| Archivo o carpeta | Contenido |
|---|---|
| [informe/](informe/) | `informe.tex` e `informe.pdf`, con el análisis, las comparaciones y las conclusiones de la misma ejecución. Pendiente. |
| [resultados/](resultados/) | Archivos generados por el experimento. Pendiente. |
| [clasificacion_simpsons_cnn.ipynb](clasificacion_simpsons_cnn.ipynb) | Código, salidas e interpretaciones. Pendiente. |
| [utilidades.py](utilidades.py) | Carga de datos, gráficos, formato de tablas y exportación. |
| [requirements.txt](requirements.txt) | Dependencias de Python. |
| [.gitignore](.gitignore) | Exclusiones de datos, pesos, entornos y archivos temporales. |