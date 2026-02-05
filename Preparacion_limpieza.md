# Preparación / Limpieza de datos

Debido a la naturaleza de los datos, como  podemos ver, tenemos ciertas variables categoricas con 2 
clases, y una tipo boleano, es complicado podemos hacer un modelo de regresion lineal multiple ya que al 
ser clases, no es algo que podamos cuantificar , de modo que tendremos que transformar los datos, un buen 
acercamiento que podriamos hacer para abarcar esta problematica es que asignaremos valores binarios para cada sexo, en 
este caso, por ejemplo, asignaremos 0 si es hombre y 1 si es mujer, de la misma manera a las demas variables, siendo 1 si 
pertenece a GP y 0 si es de MS y la misma logica para la variable de internet.

Así mismo, al tener las variables de forma numerica, podemos eliminar las variables originales, teniendo en cuenta la asignación
para cada una de las variables a transformar.

### Transformacion de variables


>Python Code


```python
# Crear nuevas variables
# para la variable sexo
df['Sexo_bin'] = np.where(df['Sexo'] == 'F',1,0)
df['Escuela_bin'] = np.where(df['Escuela'] == 'GP',1,0)
# Remplazar las categorias variable false/ true para la columna de internet
df['Internet'] = df['Internet'].map({'yes': 1, 'no': 0})
df.drop(columns=['Sexo','Escuela'],axis=1,inplace=True)
# ver cambios en el dataframe
df
```


>Output




| #   | Edad | HorasDeEstudio | Reprobadas | Internet | Faltas | G1 | G2 | G3 | Sexo_bin | Escuela_bin |
|----|------|----------------|------------|----------|--------|----|----|----|----------|-------------|
| 0  | 18   | 2              | 0          | 0        | 6      | 5  | 6  | 6  | 1        | 1           |
| 1  | 17   | 2              | 0          | 1        | 4      | 5  | 5  | 6  | 1        | 1           |
| 2  | 15   | 2              | 3          | 1        | 10     | 7  | 8  | 10 | 1        | 1           |
| 3  | 15   | 3              | 0          | 1        | 2      | 15 | 14 | 15 | 1        | 1           |
| 4  | 16   | 2              | 0          | 0        | 4      | 6  | 10 | 10 | 1        | 1           |
| ...| ...  | ...            | ...        | ...      | ...    | ...| ...| ...| ...      | ...         |
| 390| 20   | 2              | 2          | 0        | 11     | 9  | 9  | 9  | 0        | 0           |
| 391| 17   | 1              | 0          | 1        | 3      | 14 | 16 | 16 | 0        | 0           |
| 392| 21   | 1              | 3          | 0        | 3      | 10 | 8  | 7  | 0        | 0           |
| 393| 18   | 1              | 0          | 1        | 0      | 11 | 12 | 10 | 0        | 0           |
| 394| 19   | 1              | 0          | 1        | 5      | 8  | 9  | 9  | 0        | 0           |
|395 rows × 10 columns|||||||||


Ahora, revisemos si los cambios han echo que nuestras variables cambien de tener categorias a formas numericas, 
revisemos el tipo de variable que tenemos


>Python Code



```python
# ver tipos de variables
df.dtypes
```



>Output


||0|
|---|--|
|Edad|	int64|
|HorasDeEstudio|	int64|
|Reprobadas	|int64|
|Internet	|int64|
|Faltas	|int64|
|G1	|int64|
|G2	|int64|
|G3	|int64|
|Sexo_bin	|int64|
|Escuela_bin	|int64|


Bien, de esta manera, nuestros datos estan un poco mejor preparados para analizar, sabemos que queremos ver si hay 
algo que influya en la calificacion final, de modo que nuestra salida seria G3 y las demas seria las variables que
vamos a usar en nuestro dataset


No conforme con eso, veamos si hay algun tipo de huecos o valores atipicos qu pudieran complicar nuestro analisis.

### Revisar Huecos u Outliers



>Python Code



```python
# Ver si hay valores faltantes por columna
print(df.isnull().sum())
```


>Output



```text
Edad              0
HorasDeEstudio    0
Reprobadas        0
Internet          0
Faltas            0
G1                0
G2                0
G3                0
Sexo_bin          0
Escuela_bin       0
dtype: int64
```



Enhorabuena, no tenemos huecos, ahora chequemos si hay algun tipo de dato atipico o un dato extraño en nuestro dataset


### Verificar Outliers



>Python Code



```python
import pandas as pd
import matplotlib.pyplot as plt

print("Variables que faltan")
print(df.isnull().sum())
print("\nTotal de filas con algún valor faltante:", df.isnull().any(axis=1).sum())

print("Detectar outliers")
print("="*50)
print(df.describe())
```


>Output



```text
             Edad  HorasDeEstudio  Reprobadas    Internet      Faltas  \
count  395.000000      395.000000  395.000000  395.000000  395.000000   
mean    16.696203        2.035443    0.334177    0.832911    5.708861   
std      1.276043        0.839240    0.743651    0.373528    8.003096   
min     15.000000        1.000000    0.000000    0.000000    0.000000   
25%     16.000000        1.000000    0.000000    1.000000    0.000000   
50%     17.000000        2.000000    0.000000    1.000000    4.000000   
75%     18.000000        2.000000    0.000000    1.000000    8.000000   
max     22.000000        4.000000    3.000000    1.000000   75.000000   

               G1          G2          G3    Sexo_bin  Escuela_bin  
count  395.000000  395.000000  395.000000  395.000000   395.000000  
mean    10.908861   10.713924   10.415190    0.526582     0.883544  
std      3.319195    3.761505    4.581443    0.499926     0.321177  
min      3.000000    0.000000    0.000000    0.000000     0.000000  
25%      8.000000    9.000000    8.000000    0.000000     1.000000  
50%     11.000000   11.000000   11.000000    1.000000     1.000000  
75%     13.000000   13.000000   14.000000    1.000000     1.000000  
```

Bien, podemos reconocer que en este caso el outlier mas visible es el de la variable reprobadas, ya que el
valor maximo es 75, es decir, alguien tuvo 75 faltas, podria ser un error, tal vez no, pero vamos a tratarlo 
como se debe, en este caso utilizaremos el metodo de tukey, donde definiremos el rango intercuartil q1-q3 y los 
valores que se encuentren fuera de ese rango los catalogaremos como outlier o datos atipicos. Tipicamente si estos 
datos conforman un porcentaje muy pequeño del conjunto de datos, si los quitamos, no deberia de afectar mucho, y seria
lo mas convenmiente ya que al tener valores extremos, podriamos sesgar mucho el promedio y otras metricas estadisticas.


### Aplicar Metodo de Tukey



>Python Code


```python
# Aplicar método de Tukey a la variable "Faltas"
Q1 = df['Faltas'].quantile(0.25)
Q3 = df['Faltas'].quantile(0.75)
IQR = Q3 - Q1

# Límites con k=1.5 (estándar de Tukey)
k = 1.5
limite_inferior = Q1 - k * IQR
limite_superior = Q3 + k * IQR

print("="*50)
print("MÉTODO DE TUKEY - Variable: Faltas")
print("="*50)
print(f"Q1 (percentil 25): {Q1}")
print(f"Q3 (percentil 75): {Q3}")
print(f"IQR (Q3 - Q1): {IQR}")
print(f"\nCon k = {k}:")
print(f"Límite inferior: {limite_inferior}")
print(f"Límite superior: {limite_superior}")

# Identificar outliers
outliers = df[(df['Faltas'] < limite_inferior) | (df['Faltas'] > limite_superior)]
print(f"\n Total de outliers detectados: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
print(f"\nValores atípicos encontrados:")
print(outliers['Faltas'].sort_values(ascending=False).values)
```



>Output


```text
==================================================
MÉTODO DE TUKEY - Variable: Faltas
==================================================
Q1 (percentil 25): 0.0
Q3 (percentil 75): 8.0
IQR (Q3 - Q1): 8.0

Con k = 1.5:
Límite inferior: -12.0
Límite superior: 20.0

Total de outliers detectados: 15 (3.80%)

Valores atípicos encontrados:
[75 56 54 40 38 30 28 26 25 24 23 22 22 22 21]
```

Listo!, hemos identificado los valores atipicos de la variable faltas, muy probablemente haya sido un 
error de captura al momento de ingresar los datos, partiendo de eso, afortunadamente esos outliers conforman
solamente el 3.8% de nuestros datos totales, de modo que si los quitamos para que no haya mucho ruido en los datos,
no deberia repercutir en el analisis, procedamos a quitar esas revisiones.

Asi mismo, vamos a calcular nuevamente ciertas medidas estadisticas para ver como cambio nuestro dataset al quitar esos
datos.


### Quitar Outliers



>Python Code



```python
# quitar outliers

# Antes de eliminar
print(f"Tamaño original del dataset: {len(df)} filas")

# Eliminar outliers en Faltas
df_limpio = df[df['Faltas'] <= limite_superior].copy()

# Después de eliminar
print(f"Tamaño después de eliminar outliers: {len(df_limpio)} filas")
print(f"Filas eliminadas: {len(df) - len(df_limpio)} ({(len(df) - len(df_limpio))/len(df)*100:.2f}%)")

# Verificar que se eliminaron
print(f"\nNuevo máximo de Faltas: {df_limpio['Faltas'].max()}")
print(f"Nueva media de Faltas: {df_limpio['Faltas'].mean():.2f}")

# Actualizar el dataframe
df = df_limpio

print(df.describe())
```

>Output

```
Tamaño original del dataset: 395 filas
Tamaño después de eliminar outliers: 380 filas
Filas eliminadas: 15 (3.80%)

Nuevo máximo de Faltas: 20
Nueva media de Faltas: 4.60

      Edad  HorasDeEstudio  Reprobadas    Internet      Faltas  \
count  380.000000      380.000000  380.000000  380.000000  380.000000   
mean    16.671053        2.042105    0.326316    0.826316    4.602632   
std      1.274762        0.846109    0.747091    0.379337    4.968236   
min     15.000000        1.000000    0.000000    0.000000    0.000000   
25%     16.000000        1.000000    0.000000    1.000000    0.000000   
50%     17.000000        2.000000    0.000000    1.000000    3.000000   
75%     18.000000        2.000000    0.000000    1.000000    7.000000   
max     22.000000        4.000000    3.000000    1.000000   20.000000   

               G1          G2          G3    Sexo_bin  Escuela_bin  
count  380.000000  380.000000  380.000000  380.000000   380.000000  
mean    10.921053   10.723684   10.421053    0.515789     0.878947  
std      3.312896    3.772553    4.612313    0.500409     0.326618  
min      3.000000    0.000000    0.000000    0.000000     0.000000  
25%      8.000000    9.000000    8.000000    0.000000     1.000000  
50%     11.000000   11.000000   11.000000    1.000000     1.000000  
75%     13.000000   13.000000   14.000000    1.000000     1.000000  
max     19.000000   19.000000   20.000000    1.000000     1.000000  
```


Efectivamente como podemos ver, ahora nuestro valor maximo de faltas es de 20, lo cual sigue siendo alto, hablando 
relativamente, pero ahora es algo que esta dentro de nuestro rango de datos.

>Nota: Bien, ahora seria interesante crear nuevas variables a partir de unas que ya existe, in acercamiento de esto podria ser 
la variable de horas de estudio, podemos definir rangos de forma binaria, es decir, catalogarlos como, estudio bajo,
estudio moderado y estudio alto, 0 y 1 si es que aplica y esta dentro del rango




### Crear Nuevas Variables



>Python Code



```python
df['Estudio_bajo'] = np.where(df['HorasDeEstudio'] == 1, 1, 0)

df['Estudio_moderado'] = np.where((df['HorasDeEstudio'] >= 2) & (df['HorasDeEstudio'] <= 3), 1, 0)

df['Estudio_alto'] = np.where(df['HorasDeEstudio'] == 4, 1, 0)

```



>Output


| # | Edad | HorasDeEstudio | Reprobadas | Internet | Faltas | G1 | G2 | G3 | Sexo_bin | Escuela_bin | Estudio_bajo | Estudio_moderado | Estudio_alto |
|---|------|----------------|------------|----------|--------|----|----|----|----------|-------------|--------------|------------------|--------------|
| 0 | 18   | 2              | 0          | 0        | 6      | 5  | 6  | 6  | 1        | 1           | 0            | 1                | 0            |
| 1 | 17   | 2              | 0          | 1        | 4      | 5  | 5  | 6  | 1        | 1           | 0            | 1                | 0            |
| 2 | 15   | 2              | 3          | 1        | 10     | 7  | 8  | 10 | 1        | 1           | 0            | 1                | 0            |
| 3 | 15   | 3              | 0          | 1        | 2      | 15 | 14 | 15 | 1        | 1           | 0            | 1                | 0            |
| 4 | 16   | 2              | 0          | 0        | 4      | 6  | 10 | 10 | 1        | 1           | 0            | 1                | 0            |


Nota: en este caso no eliminamos la variable original de horas de estudio ya 
que nos sirve como referencia a las demas, por lo tanto la dejaremos dentro del dataset


Bien, con esto ya tenemos una base solida para empezar a trabajar nuestros datos, en el siguiente modulo exploraremos 
posibles problemas de correlación de las variables que tenemos.
