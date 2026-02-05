# Exploracion de datos
En este apartado exploraremos el conjunto de datos, como esta conformado y contextualizaremos los datos para un mejor entendimiento.

## Carga de Datos
Carguemos el dataset para ver que variables tenemos


> Python Code

```python
from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')
# Ver dataset
df = pd.read_csv('/content/drive/MyDrive/Inteligencia_Artificial_1/A1.3 Calificaciones.csv')
df
```

>Output


| #   | Escuela | Sexo | Edad | HorasDeEstudio | Reprobadas | Internet | Faltas | G1 | G2 | G3 |
|----|---------|------|------|----------------|------------|----------|--------|----|----|----|
| 0  | GP      | F    | 18   | 2              | 0          | no       | 6      | 5  | 6  | 6  |
| 1  | GP      | F    | 17   | 2              | 0          | yes      | 4      | 5  | 5  | 6  |
| 2  | GP      | F    | 15   | 2              | 3          | yes      | 10     | 7  | 8  | 10 |
| 3  | GP      | F    | 15   | 3              | 0          | yes      | 2      | 15 | 14 | 15 |
| 4  | GP      | F    | 16   | 2              | 0          | no       | 4      | 6  | 10 | 10 |
| ...| ...     | ...  | ...  | ...            | ...        | ...      | ...    | ...| ...| ...|
| 390| MS      | M    | 20   | 2              | 2          | no       | 11     | 9  | 9  | 9  |
| 391| MS      | M    | 17   | 1              | 0          | yes      | 3      | 14 | 16 | 16 |
| 392| MS      | M    | 21   | 1              | 3          | no       | 3      | 10 | 8  | 7  |
| 393| MS      | M    | 18   | 1              | 0          | yes      | 0      | 11 | 12 | 10 |
| 394| MS      | M    | 19   | 1              | 0          | yes      | 5      | 8  | 9  | 9  |
|395 rows × 10 columns||||||||||

El conjunto de datos está compuesto por 395 observaciones y 10 variables de interés. Entre ellas se encuentran variables numéricas y categóricas.

Se aprecian dos variables categóricas principales (Escuela y Sexo), además de una variable que representa una condición binaria (Internet), la cual está almacenada como texto ("yes"/"no") y no como tipo booleano nativo. Este tipo de variables requiere transformación para poder ser utilizadas en modelos estadísticos.

Las variables categóricas pueden representar un desafío, ya que los modelos de regresión trabajan con valores numéricos. Por lo tanto, será necesario aplicar técnicas de codificación en la etapa de limpieza de datos.

Según las fuentes originales del conjunto de datos:

- Las calificaciones (G1, G2, G3) se encuentran en una escala de 0 a 20, distinta a la escala utilizada comúnmente en México.

- Las horas de estudio se miden de forma semanal.

- Las categorías GP y MS corresponden a las instituciones "Gabriel Pereira" y "Mousinho da Silveira"

Este contexto es importante para interpretar adecuadamente los resultados posteriores del análisis.

