
# Solución de Problemas
En este modulo abarcaremos ciertos problemas o complicaciones que podriamos encontrar al trabajar en una base de datos, tales como colinearidad, outliers, huecos, entre otros.

🔹 Outliers (valores atípicos): Datos que se alejan mucho del resto. Pueden alterar promedios y distorsionar modelos.

🔹 Colinealidad: Variables independientes muy correlacionadas entre sí. Dificulta saber qué variable realmente influye en el modelo.

🔹 Huecos o datos faltantes: Información no registrada. Puede generar sesgos si no se trata adecuadamente.

🔹 Ruido: Variaciones aleatorias o errores que no representan un patrón real y reducen la precisión del análisis.

### Recursos

|Dataset|[Calificaciones.csv](Calificaciones.csv)|
|-------|--------|
|Notebook|[.ipynb](Tarea_A1_aprendizaje_estadistico_automatico.ipynb)|

Esto con el objetivo de aprender a lidiar con estos tipos de situaciones y poder manipular el dataset para obtener un resultado estadisticamente confiable.

## Metodología
- Carga y Exploración de Datos: explorar la cantidad de observaciones, escalas y variables que tenemos
- Preparación y Limpieza: limpiar datos, llenar huecos, identificar outliers.
- Análisis de posibles relaciones entre variables: análisis mediante matriz de correlación e interpretación
- Selección de características: selección de variables que aportan y cuáles no
- Entrenamiento y Evaluación del modelo: creación de un modelo de regresión lineal múltiple
- Conclusión: conclusión general de esta práctica.
# Procedimiento

[Preparación / Limpieza](Data_Exploring.md)

[Diseño del Estudio](Study_Design.md)

[Relaciones entre variables y seleccion de caracteristicas](Output_variable.md)

[Conclusiones](Grafic_analysis_conclusion.md)
