# Análisis de objetos en video

Mini librería para registrar detecciones de objetos y proveer una interfaz para obtener información de las detecciones.

## Instalación

El proyecto requiere python 3 y la instalación de las dependencias de especificadas en el archivo `requirements.txt` 

```
pip install -r requirements.txt
```

## Demo

Para correr la aplicación de demostración se debe correr el siguiente comando.

```
streamlit run main.py
```

## Detección de objetos

El archivo `object_detector.py` contiene una clase ObjectDetector que procesa los frames de video con una red neuronal (Darknet Neural Network) a través de los métodos `stream_videofile` y `stream_webcam`. El valor de retorno de estos es una colección de registros.

## Análisis

El archivo `video_analysis.py` contiene una clase VideoAnalysis que procesa un pandas `DataFrame` con los registros obtenidos de ObjectDetector. 

### Métodos

#### get_unique_classes(start_time=0, end_time=-1, as_seconds=False)

Obtiene las clases encontradas en el video en el rango [start_time, end_time] del video. 

##### argumentos

| Nombre | Tipo | Descripción |
| --- | --- | --- |
| `as_seconds` | booleano | indica si los rangos deben considerarse como segundos. Por defecto se usan milisegundos. |
| `star_time` | entero | El limite inferior de tiempo para buscar.|
| `end_time` | entero | El limite superior de tiempo para buscar. -1 indica que se buscará hasta el final del video. |


### get_timeranges_with_classes(target_classes, time_tolerance = 2000)

Obtiene una lista de rangos de tiempo en los que aparecen `target_classes`.

##### argumentos

| Nombre | Tipo | Descripción |
| --- | --- | --- |
| `target_classes` | string[] | Lista de clases de objetos a buscar en el video |
| `time_tolerance` | entero | Es la tolerancia en ms para que dos detecciones se consideren dentro de un mismo rango de tiempo. |

### get_timeranges_by_instance_counts(class_counts, time_tolerance = 2000):

Obtiene una lista de rangos de tiempo en los que aparecen las clases en `class_counts` con su respectiva cantidad de repeticiones.

##### argumentos

| Nombre | Tipo | Descripción |
| --- | --- | --- |
| `class_counts` | dict | Diccionario con repeticiones de clases. Mapea nombres de clases a el número de instancias que se deben buscar de esa clase |
| `time_tolerance` | entero | Es la tolerancia en ms para que dos detecciones se consideren dentro de un mismo rango de tiempo. |