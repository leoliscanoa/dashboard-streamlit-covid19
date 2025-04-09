# COVID-19 Data Warehouse Dashboard 🦠

Este proyecto es un **dashboard interactivo desarrollado con Streamlit** para monitorear y analizar datos relacionados con el COVID-19. Combina datos desde múltiples fuentes (API y CSV) y los estructura en un almacén local para visualización y análisis.

## Funcionalidades Principales 🚀

- **Extracción de datos desde múltiples fuentes**:
  - API de COVID-19 ([API URL](https://coronavirus.m.pipedream.net/)).
  - Archivos CSV locales.

- **Transformación de los datos**:
  - Limpieza automática de columnas y datos faltantes.
  - Formateo de nombres de columnas.
  - Eliminación de datos duplicados.

- **Carga a un almacén de datos local**:
  - Almacenamiento en una base de datos SQLite para consultas rápidas.
  - Dos tablas principales:
    - `covid_patients`: Datos de pacientes desde CSV.
    - `covid_country_stats`: Estadísticas por país desde la API.

- **Interfaz interactiva**:
  - Filtros avanzados para explorar los datos.
  - Visualizaciones dinámicas con Plotly para análisis descriptivo.

## Requisitos Previos 🛠️

- Python ≥ 3.8
- Paquetes instalables mediante pip:
  - `streamlit`
  - `pandas`
  - `requests`
  - `sqlite3` (módulo estándar de Python)
  - `plotly`

## Instalación 🔧

1. Clona este repositorio en tu máquina local:
   ```bash
   git clone https://github.com/leoliscanoa/dashboard-streamlit-covid19.git
   cd dashboard-streamlit-covid19
   ```

2. Instala las dependencias requeridas:
   ```bash
   pip install -r requirements.txt
   ```

## Uso ▶️
1. Ejecuta el script principal:
   ```bash
   streamlit run app.py
   ```

2. Accede al dashboard en tu navegador en la URL: [http://localhost:8501](http://localhost:8501).

3. Explora las estadísticas del COVID-19 utilizando las herramientas de filtrado y visualización disponibles.

## Estructura del Proyecto 📂

- `app.py`: Script principal que gestiona el flujo de trabajo ETL (Extracción, Transformación y Carga) y renderiza el dashboard.
- `data/`: Directorio para almacenar archivos CSV necesarios.
- `cache/`: Carpeta temporal para almacenamiento en caché de datos provenientes de la API.
- `covid_warehouse.db`: Base de datos SQLite generada al cargar los datos.

## Flujo ETL (Extract, Transform, Load) 🔄

1. **EXTRACCIÓN**:
   - Los datos se recopilan desde un archivo CSV y una API externa.
   - Uso de caché para minimizar llamadas innecesarias a la API.

2. **TRANSFORMACIÓN**:
   - Limpieza de datos (nombres de columnas, valores nulos, datos duplicados).
   - Conversión de tipos y valores booleanos.

3. **CARGA**:
   - Inserción de los datos en una base SQLite organizada en tablas.

## Visualizaciones 📊

El proyecto utiliza **Plotly** para generar gráficos dinámicos que permiten analizar:

- Casos confirmados y fallecidos por región.
- Tasas de incidencia y letalidad por país.
- Tendencias temporales del COVID-19.