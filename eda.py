import pandas as pd
from pandas import DataFrame
import logging
#   config básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#   Cargamos el dataset
df: DataFrame = pd.read_csv('./data/nashville_accidents_data.csv')

#   ======================================================================
#   Exploramos un poco el dataset antes de proceder a limpiarlo
#   ======================================================================
#   (filas, columnas)
logging.info(df.shape)

#   columnas y tipos
logging.info(df.dtypes)
logging.info(df.info())

#   valores nulos por columna
logging.info(df.isnull().sum())

#   estadísticas descriptivas
logging.info(df.describe())

#   Primeros 5 registros del dataset
logging.info(df.head())
#   ======================================================================
#   Limpieza de datos y transformación del dataset
#   ======================================================================

#   1. Estandarizar los nombres de las columnas
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace(r'[^a-z0-9_]', '', regex=True)
)
#   logging.info(df.columns)

#   2. Conversión de fechas y extracción de año, mes y día
df['date_and_time'] = pd.to_datetime(df['date_and_time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

df['year'] = df['date_and_time'].dt.year
df['month'] = df['date_and_time'].dt.month
df['hour'] = df['date_and_time'].dt.hour
df['weekday'] = df['date_and_time'].dt.day_name()

#   logging.info(df.dtypes)

#   3. Eliminación de columnas innecesarias
#   Aunque parece que eliminamos muchas columnas, en realidad son columnas que no aportan valor al análisis
columnas_a_eliminar: list = [
    'accident_number',
    'property_damage',
    'street_address',
    'city',
    'state',
    'harmfulcodes',
    'harmfuldescriptions',
    'objectid',
    'zip_code',
    'rpa',
    'weather',
    'illuaccidemination',
    'collision_type',
    'reporting_officer',
    'x',
    'y'
]
df.drop(columns=columnas_a_eliminar, inplace=True, errors='ignore')
logging.info(df.dtypes)

#   4. Tratamiento de valores nulos

#| Columna                      | Tipo       | Estrategia recomendada                          |
#| ---------------------------- | ---------- | ----------------------------------------------- |
#| `number_of_motor_vehicles`   | Numérica   | Rellenar con la **mediana**                     |
#| `number_of_injuries`         | Numérica   | Rellenar con **0** (accidente sin heridos)      |
#| `number_of_fatalities`       | Numérica   | Rellenar con **0**                              |
#| `hit_and_run`                | Categórica | Rellenar con "UNKNOWN"                     |
#| `collision_type_description` | Categórica | `"UNSPECIFIED"`                             |
#| `weather_description`        | Categórica | `"UNKNOWN"`                                     |
#| `illumination_description`   | Categórica | `"UNKNOWN"`                                     |
#| `precinct`                   | Categórica | `"UNKNOWN"`                                     |
#| `lat`, `long`                | Numérica   | **Eliminar filas** si no hay geolocalización    |


# Numéricas
df['number_of_motor_vehicles'] = df['number_of_motor_vehicles'].fillna(df['number_of_motor_vehicles'].median())
df['number_of_injuries'] = df['number_of_injuries'].fillna(0)
df['number_of_fatalities'] = df['number_of_fatalities'].fillna(0)

# Categóricas
df['hit_and_run'] = df['hit_and_run'].fillna('UNKNOWN')
df['collision_type_description'] = df['collision_type_description'].fillna('UNSPECIFIED')
df['weather_description'] = df['weather_description'].fillna('UNKNOWN')
df['illumination_description'] = df['illumination_description'].fillna('UNKNOWN')
df['precinct'] = df['precinct'].fillna('UNKNOWN')

# Geolocalización: eliminar filas sin lat/lon
df = df.dropna(subset=['lat', 'long'])

# Verificamos que no hay valores nulos en las columnas que nos interesan
logging.info(df[['number_of_motor_vehicles', 'number_of_injuries', 'number_of_fatalities',
          'hit_and_run', 'collision_type_description', 'weather_description',
          'illumination_description', 'precinct']].isnull().sum())

#   ======================================================================
#   Analisis exploratorio de datos (EDA)
#   ======================================================================

#   ¿En qué días y horarios ocurren más accidentes?

import seaborn as sns
import matplotlib.pyplot as plt

dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='weekday', order=dias)
plt.title("Cantidad de accidentes por día de la semana")
plt.xlabel("Día")
plt.ylabel("Cantidad")
plt.tight_layout()
plt.show()

#   Accidentes por hora del día

plt.figure(figsize=(10, 4))
sns.histplot(df['hour'], bins=24, kde=True)
plt.title("Distribución horaria de los accidentes")
plt.xlabel("Hora del día")
plt.ylabel("Cantidad de accidentes")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()

#   ¿Cuál es la relación entre las condiciones climáticas y la ocurrencia de accidentes?

top_weather = df['weather_description'].value_counts().head(7)

plt.figure(figsize=(8, 4))
sns.barplot(x=top_weather.values, y=top_weather.index)
plt.title("Condición climática al momento del accidente")
plt.xlabel("Cantidad")
plt.ylabel("Condición climática")
plt.tight_layout()
plt.show()

#   ¿Qué tipo de accidentes son más frecuentes y qué factores los acompañan?

top_collision = df['collision_type_description'].value_counts().head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x=top_collision.values, y=top_collision.index)
plt.title("Tipos de colisión más frecuentes")
plt.xlabel("Cantidad")
plt.ylabel("Tipo de colisión")
plt.tight_layout()
plt.show()

#   ¿Hay diferencias notorias entre estaciones del año?

def obtener_estacion(mes):
    if mes in [12, 1, 2]:
        return 'WINTER'
    elif mes in [3, 4, 5]:
        return 'SPRING'
    elif mes in [6, 7, 8]:
        return 'SUMMER'
    else:
        return 'AUTUMN'

df['month'] = df['date_and_time'].dt.month
df['estacion'] = df['month'].apply(obtener_estacion)

plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='estacion', order=['WINTER', 'SPRING', 'SUMMER', 'AUTUMN'])
plt.title("Accidentes por estación del año")
plt.xlabel("Estación")
plt.ylabel("Cantidad")
plt.tight_layout()
plt.show()

#   Accidentes por condición climática y hora del día
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='weather_description', y='hour')
plt.xticks(rotation=45)
plt.title("Hora del accidente según condición climática")
plt.xlabel("Clima")
plt.ylabel("Hora del día")
plt.tight_layout()
plt.show()