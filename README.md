![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

# Análisis de Accidentes de Tránsito en Nashville (2018–2025)

Este proyecto corresponde al **Trabajo Práctico de la Entrega 1** del curso de Data Science II de Coderhouse, realizado por **Pedro Díaz**. El objetivo fue explorar, limpiar y analizar un dataset real de accidentes de tránsito en la ciudad de Nashville para obtener insights que puedan ser útiles para la planificación urbana y la seguridad vial.

---

## 📜 Contenido del repositorio

```
datascienceII-coderhouse/
│
├── eda.py                           # Script de análisis exploratorio (EDA)
├── Informe TP Entrega 1 - Pedro Diaz.pdf  # Informe final con visualizaciones e insights
├── requirements.txt                # Librerías necesarias para correr el proyecto
├── README.md                       # Este archivo
├── .gitignore                      # Exclusiones del repositorio
├── LICENSE                         # Licencia MIT
└── data/
    └── nashville_accidents_data.csv  # Dataset original utilizado
```

---

## 🧠 Abstracto

El análisis busca comprender los factores asociados a los accidentes viales en Nashville entre 2018 y 2025. Se estudian patrones temporales, geográficos, climáticos y de severidad, con el fin de responder preguntas clave y generar visualizaciones útiles para la toma de decisiones.

---

## 📌 Preguntas / Hipótesis abordadas

1. ¿En qué días y horarios ocurren más accidentes?
2. ¿Existen zonas con mayor concentración geográfica de accidentes?
3. ¿Cuál es la relación entre el clima y la ocurrencia de accidentes?
4. ¿Qué tipos de colisiones son más frecuentes?
5. ¿Se redujeron los accidentes tras el inicio de la pandemia?
6. ¿Existen diferencias estacionales?
7. ¿Se reportan más accidentes en intersecciones o en vías rectas?

---

## 📈 Visualizaciones

El análisis incluye gráficos como:

* Accidentes por día de la semana y hora del día
* Condiciones climáticas y tipo de colisión
* Distribución por estación del año
* Gráficos multivariados para entender la severidad

Ver informe completo en [Informe TP Entrega 1 - Pedro Diaz.pdf](./Informe%20TP%20Entrega%201%20-%20Data%20Science%20-%20Pedro%20Diaz.pdf)

---

## ⚙️ Requisitos

Instalá las dependencias con:

```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecución

Podés ejecutar el análisis directamente con:

```bash
python eda.py
```

También podés utilizar Google Colab si preferís un entorno interactivo.

---
