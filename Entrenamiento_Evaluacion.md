# Entrenamiento y evaluación del modelo

Como vimos en el modulo de limpieza y selección de caracteristicas, exploremos los dos escenarios donde
incluyamos un modelo donde incluyamos las calificaciones del primer y segundo periodo y uno donde no las incluyamos, 
esto para ver como varian nuestras metricas estadisticas y nuestros modelos bajo distintas condiciones.


## Regresión Lineal Multiple CON g1 y g2
Para estos dos datos, usemos el 70% de los datos originales ya que en si nuestro dataset de 380 observaciones, es
relativamente pequeño, de modo que hay que balancearlo, para el training de los dos casos, obteniendo datos como el pvalue, RSE, 
RSS, F statistic, antes y despues del training para ver como se comporta.

### Importar librerias y partir datos en train y test

>Python Code

```python
# Genera datos de entrenamiento
train = df.sample(frac = 0.7)
# Genera datos de validación
test = df.drop(train.index)
# Imprime dimensiones de datos de entrenamiento
print("Train:", train.shape)
# Imprime dimensiones de datos de prueba
print("Test:",test.shape)
# Imprime primeras 5 filas de datos de entrenamiento
print(train.head())

```



>Output



```text
Train: (266, 13)
Test: (114, 13)
     Edad  HorasDeEstudio  Reprobadas  Internet  Faltas  G1  G2  G3  Sexo_bin  \
206    16               2           3         1       5   7   7   7         1   
45     15               2           0         1       8   8   8   6         1   
349    18               1           1         1      10  11  13  13         0   
5      16               2           0         1      10  15  15  15         0   
244    18               3           0         1       0   7   0   0         1   

     Escuela_bin  Estudio_bajo  Estudio_moderado  Estudio_alto  
206            1             0                 1             0  
45             1             0                 1             0  
349            0             1                 0             0  
5              1             0                 1             0  
244            1             0                 1             0
```



### Entrenar Modelo


>Python Code



```python
# Importar librería
import statsmodels.api as sm
# Generar elemento X
X = train.drop('G3', axis = 1)
# Generar elemento Y
Y = train.G3
# Definir el tipo de modelo
model = sm.OLS(Y,sm.add_constant(X))
# Ajustar el modelo para obtener resultados
results = model.fit()
# Imprimir un resumen de los resultados
print(results.summary())
```


>Output


```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                     G3   R-squared:                       0.845
Model:                            OLS   Adj. R-squared:                  0.838
Method:                 Least Squares   F-statistic:                     125.8
Date:                Thu, 05 Feb 2026   Prob (F-statistic):           4.78e-96
Time:                        01:44:20   Log-Likelihood:                -543.73
No. Observations:                 266   AIC:                             1111.
Df Residuals:                     254   BIC:                             1154.
Df Model:                          11                                         
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
const                0.7331      1.563      0.469      0.639      -2.345       3.811
Edad                -0.2207      0.112     -1.974      0.049      -0.441      -0.001
HorasDeEstudio       0.3776      0.346      1.090      0.277      -0.305       1.060
Reprobadas          -0.4943      0.185     -2.671      0.008      -0.859      -0.130
Internet            -0.3600      0.323     -1.116      0.266      -0.995       0.275
Faltas               0.1197      0.024      4.906      0.000       0.072       0.168
G1                   0.1793      0.069      2.581      0.010       0.042       0.316
G2                   0.9475      0.060     15.790      0.000       0.829       1.066
Sexo_bin            -0.1572      0.257     -0.610      0.542      -0.664       0.350
Escuela_bin         -0.2535      0.414     -0.612      0.541      -1.070       0.562
Estudio_bajo         1.1770      0.647      1.819      0.070      -0.097       2.451
Estudio_moderado     0.5465      0.557      0.981      0.327      -0.551       1.644
Estudio_alto        -0.9904      0.891     -1.111      0.268      -2.746       0.765
==============================================================================
Omnibus:                      122.124   Durbin-Watson:                   1.969
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              525.348
Skew:                          -1.911   Prob(JB):                    8.36e-115
Kurtosis:                       8.726   Cond. No.                     1.30e+17
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 8.75e-30. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
```

Enhorabuena, vemos que tenemos una R^2 ajustada muy alta, de 0.838, lo cual nos dice que nuestro modelo explica el 83.8% del
conjunto de datos que tenemos, asi mismo podemos notar que hay ciertas variables las cuales debido a su pvalue, es probable 
que no aporten mucho al dataset, tales como, 'Escuela_bin', 'Sexo_bin'.


### Interpretacion de la tabla
De esta tabla de datos podemos interpretar lo siguiente:

- Edad: Un coeficiente negativo indica que, a mayor edad, la calificación tiende a bajar ligeramente.

- Repetición y acceso a internet: Si repites el año y tienes internet, eso también afecta negativamente la calificación.

- Escuela y sexo: La variable de sexo, con un coeficiente negativo para mujeres, muestra una diferencia de calificación, aunque pequeña.

- Horas de estudio: Lo interesante es que, en este caso, estudiar más no necesariamente mejora la calificación; al contrario, podría estar relacionado con una menor calificación.

Así mismo, podemos ver que las variables que practicamente tiene un p-val de 0, son faltas (algo extraño de encontrar) y G2, lo que nos
pued indicar que estas variables son importantes en nuestro dataset.



### Metricas Estadisticas (F-stat / p-val)



>Python Code



```python
import scipy.stats as st
yhat = results.predict(sm.add_constant(X))
ybar = np.mean(Y)
ESS = sum((yhat - ybar)**2)
m = X.shape[1]
EMS = ESS / m
RSS = sum((Y - yhat)**2)
n = X.shape[0]
RMS = RSS / (n - m - 1)
F = EMS / RMS
pval = st.f.sf(F, m, n - m - 1)
print("F =", F)
print("p-value =", pval)
```


>OutPut


```text
F = 114.83046878759362
p-value = 5.490568541948647e-95
```

Esto nos dice que en nuestro modelo, hay variables importantes y debido al pvalue pequeño,
la probabilidad de que esto sea al azar es bajisima, cosa que nos indica que vamos por buen camino.

Nota: Como las variables, Escuela_bin y Sexo_bin,  tirenen un pvalue de casi 50%, vamos a quitarlas y a definir un nuevo dataset sin esas variables


### Eliminar variables con p-val muy grande



>Python Code



```python
# generar nuevos modelos sin las variables con un pvalue muy grande
XNew = X.drop(['Escuela_bin','Sexo_bin'], axis = 1)
modelNew = sm.OLS(Y,sm.add_constant(XNew))
resultsNew = modelNew.fit()
```


Calcular metricas estadisticas para esas variables, para posteriormente usarlas 


>Python Code


```python
yhatNew = resultsNew.predict(sm.add_constant(XNew))
RSSNew = sum((Y-yhatNew)**2)
EMSNew = (RSSNew - RSS) / 1
FNew = EMSNew / RMS
pvalNew = st.f.sf(FNew, 1, n-m-1)
t = np.sqrt(FNew)
print("New F =", FNew)
print("t-value =", t)
print("p-value =", pvalNew)
print("OLS's p-value Escuela =", results.pvalues["Escuela_bin"])
print("OLS's p-value Sexo =", results.pvalues["Sexo_bin"])
```


>Output



```text
New F = 0.7352437702228594
t-value = 0.8574635678691307
p-value = 0.392000256699059
OLS's p-value Escuela = 0.5411805124962055
OLS's p-value Sexo = 0.5421269209145818
```


### Generar Nuevos elementos para la validacion



>Python Code


```python
# Genera el elemento XTest
XTest = test.drop('G3', axis = 1)
yhatTest = results.predict(sm.add_constant(XTest))
YTest = test.G3
RSSTest = sum((YTest-yhatTest)**2)
TSSTest = sum((YTest-np.mean(YTest))**2)
nTest = XTest.shape[0]
mTest = XTest.shape[1]
RSETest = np.sqrt(RSSTest/(nTest))
R2Test = 1 - RSSTest / TSSTest
print("RSE =", RSETest)
print("R^2 =", R2Test)
```



>Output



```text
RSE = 1.8853513045702408
R^2 = 0.8046668364678251
```

Vemos que tenemos un R^2 de 0.8, es decir, nuestro modelo explica el 80% de los datos del dataset, un buen numero hablando relativamente, y con un RSE de 1.88 de variacion en cuanto a la salida:
