#Link del video
https://www.youtube.com/watch?v=60VzlVZXMxQ

# Predicción de Abandono Estudiantil

Este proyecto busca simular y predecir la *deserción estudiantil en el primer año académico* mediante técnicas de *Machine Learning*.  
Se generó un dataset sintético de 500 registros que contiene información *demográfica, académica y financiera* de estudiantes, junto con la variable objetivo: *Abandono (Sí/No)*.  

---

## Dataset Sintético

El dataset fue generado en Python e incluye las siguientes variables:

###  Variables demográficas
- *Edad* (numérica): entre 16 y 30 años.  
- *Género* (categórica): Masculino, Femenino, Otro.  
- *Origen* (categórica): Urbano, Rural.  

###  Variables académicas
- *Promedio_Colegio* (numérica): promedio de notas de colegio (0 a 5, con algunos valores atípicos como -1 o 10).  
- *Examen_Admision* (numérica): puntaje entre 0 y 100, con outliers incluidos (-50 y 200).  
- *Promedio_Primer_Semestre* (numérica): desempeño en la universidad, escala 0 a 5.  

### Variables financieras
- *Nivel_Socioeconomico* (categórica): Bajo, Medio, Alto.  
- *Beca* (binaria): 0 = No tiene, 1 = Sí tiene.  
- *Prestamo* (binaria): 0 = No tiene, 1 = Sí tiene.  

### Variable objetivo
- *Abandono* (binaria): 0 = No abandona, 1 = Sí abandona.  

⚠ El dataset contiene *valores nulos y atípicos* para simular un escenario realista.  

---

## Modelo de Aprendizaje Automático

Se utilizó un modelo de *Regresión Logística*, ya que:  
- Es un problema de *clasificación binaria*.  
- El modelo es *sencillo, rápido e interpretable*.  
- Permite identificar qué variables influyen más en el riesgo de abandono.  

*Funcionamiento:*  
El modelo calcula una probabilidad entre 0 y 1 de que el estudiante abandone.  
Si la probabilidad es mayor a 0.5 → predice *abandono*.  
Si es menor → predice *continúa*.  

---

## Requisitos

Instalar las dependencias con:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn