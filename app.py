import os
import json
import sqlite3
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta

# Configuración de la página para maximizar el ancho
st.set_page_config(
    page_title="COVID-19 Data Warehouse Dashboard",
    page_icon="🦠",
    layout="wide",  # Usar todo el ancho disponible
    initial_sidebar_state="collapsed"  # Colapsar sidebar por defecto
)

# Configuración general
os.makedirs("cache", exist_ok=True)
API_URL = "https://coronavirus.m.pipedream.net/"
CACHE_FILE = "cache/covid_api_cache.json"
CACHE_TTL = timedelta(minutes=30)
NOW = datetime.now()


# Funciones auxiliares
def fetch_api_data():
    if os.path.exists(CACHE_FILE):
        modified = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
        if NOW - modified < CACHE_TTL:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
        else:
            response = requests.get(API_URL)
            data = response.json()
            with open(CACHE_FILE, "w") as f:
                json.dump(data, f)
    else:
        response = requests.get(API_URL)
        data = response.json()
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f)
    return data


def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def clean_boolean_columns(df):
    for column in df.columns:
        if df[column].nunique() <= 3 and df[column].dropna().isin([1, 2]).all():
            df[column] = df[column].replace({1: True, 2: False}).infer_objects(copy=False).astype('boolean')
    return df


def clean_missing_data(df):
    df = df.replace({97: 'not_defined', 99: 'not_defined'})
    df = df.fillna('not_defined')
    return df


def clean_drop_duplicates(df):
    return df.drop_duplicates()


def load_data():
    # ETL Process
    # --- EXTRACT ---
    # CSV
    with st.spinner("Extrayendo datos del CSV..."):
        df_csv = pd.read_csv("./data/covid_data.csv")

    # API
    with st.spinner("Extrayendo datos de la API..."):
        api_data = fetch_api_data()

    # --- TRANSFORM ---
    with st.spinner("Transformando datos..."):
        df_csv = clean_column_names(df_csv)
        df_csv = clean_boolean_columns(df_csv)
        df_csv = clean_missing_data(df_csv)
        df_csv = clean_drop_duplicates(df_csv)

        raw_data = api_data.get("rawData", [])
        df_api = pd.DataFrame(raw_data)
        df_api = clean_column_names(df_api)
        df_api = clean_drop_duplicates(df_api)
        cols_to_keep = ['country_region', 'confirmed', 'deaths', 'lat', 'long_', 'incident_rate', 'case_fatality_ratio']
        df_api = df_api[cols_to_keep]
        for col in ['confirmed', 'deaths']:
            df_api[col] = pd.to_numeric(df_api[col], errors='coerce').fillna(0).astype(int)
        for col in ['incident_rate', 'case_fatality_ratio']:
            df_api[col] = pd.to_numeric(df_api[col], errors='coerce')

    # --- LOAD ---
    with st.spinner("Cargando datos en la base de datos..."):
        conn = sqlite3.connect("covid_warehouse.db")
        df_csv.to_sql("covid_patients", conn, if_exists="replace", index=False)
        df_api.to_sql("covid_country_stats", conn, if_exists="replace", index=False)

    st.toast("¡Datos cargados correctamente!", icon="✅")
    return conn, df_csv, df_api


def filter_dataframe(df, key_prefix):
    """
    Agrega controles de UI para filtrar un dataframe por columnas.
    """
    with st.expander("Filtros avanzados", expanded=False):
        # Limitar a 10 filtros para no sobrecargar la interfaz
        display_columns = df.columns[:10] if len(df.columns) > 10 else df.columns

        # Crear columnas para organizar los filtros
        cols = st.columns(3)
        filters = {}

        # Recorrer las columnas y crear filtros apropiados según el tipo de datos
        for i, col in enumerate(display_columns):
            col_idx = i % 3
            with cols[col_idx]:
                if df[col].dtype == 'object' or df[col].dtype == 'boolean' or df[col].dtype == 'category':
                    # Para columnas categóricas o strings
                    unique_values = df[col].unique()
                    if len(unique_values) <= 10:  # Solo mostrar multiselect si hay pocas opciones
                        selected_values = st.multiselect(
                            f"Valores para {col}",
                            options=unique_values,
                            default=[],
                            key=f"{key_prefix}_{col}"
                        )
                        if selected_values:
                            filters[col] = selected_values
                    else:
                        search_term = st.text_input(
                            f"Buscar en {col}",
                            "",
                            key=f"{key_prefix}_{col}"
                        )
                        if search_term:
                            filters[col] = search_term
                elif pd.api.types.is_numeric_dtype(df[col]):
                    # Para columnas numéricas
                    min_val, max_val = float(df[col].min()), float(df[col].max())
                    range_selected = st.slider(
                        f"Rango para {col}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        key=f"{key_prefix}_{col}"
                    )
                    if range_selected != (min_val, max_val):
                        filters[col] = range_selected
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    # Para columnas de fecha/hora
                    min_date, max_date = df[col].min(), df[col].max()
                    date_selected = st.date_input(
                        f"Fecha para {col}",
                        value=(min_date, max_date),
                        key=f"{key_prefix}_{col}"
                    )
                    if date_selected != (min_date, max_date):
                        filters[col] = date_selected

        # Aplicar los filtros al dataframe
        filtered_df = df.copy()
        for col, filter_val in filters.items():
            if isinstance(filter_val, list):
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Filtro de rango numérico
                    filtered_df = filtered_df[(filtered_df[col] >= filter_val[0]) & (filtered_df[col] <= filter_val[1])]
                else:
                    # Filtro de selección múltiple
                    filtered_df = filtered_df[filtered_df[col].isin(filter_val)]
            elif isinstance(filter_val, str):
                # Filtro de texto
                filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(filter_val, case=False)]
            elif isinstance(filter_val, tuple) and len(filter_val) == 2:
                # Filtro de rango (numérico o fecha)
                filtered_df = filtered_df[(filtered_df[col] >= filter_val[0]) & (filtered_df[col] <= filter_val[1])]

    return filtered_df


def main():
    st.title("🦠 COVID-19 Data Warehouse Dashboard")
    st.markdown("### Sistema de análisis de datos integrado")
    st.markdown("### Autor: Leo Liscano")

    # Cargar datos
    conn, df_patients, df_country_stats = load_data()

    # Crear pestañas para organizar mejor el contenido
    tab1, tab2, tab3 = st.tabs(["Bases de datos", "Análisis dinámico", "Análisis estático"])

    with tab1:
        st.header("Exploración de datos básicos")

        # Organizar las consultas simples en columnas (una al lado de la otra)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Datos por país")

            # Control para seleccionar número de registros a mostrar
            num_records_country = st.number_input(
                "Número de registros a mostrar",
                min_value=1,
                max_value=100,
                value=10,
                key="num_records_country"
            )

            # Obtener todos los datos para aplicar filtros
            query_country = pd.read_sql_query(f"SELECT * FROM covid_country_stats", conn)

            # Aplicar filtros interactivos a los datos de países
            filtered_country_data = filter_dataframe(query_country, "country")

            # Mostrar los datos filtrados limitados por el número seleccionado
            st.dataframe(filtered_country_data.head(num_records_country), use_container_width=True)

            # Mostrar estadísticas básicas de los datos filtrados
            with st.expander("📈 Estadísticas de los datos filtrados por país", expanded=True):
                st.write(filtered_country_data.describe())

        with col2:
            st.subheader("👥 Datos de pacientes")

            # Control para seleccionar número de registros a mostrar
            num_records_patients = st.number_input(
                "Número de registros a mostrar",
                min_value=1,
                max_value=100,
                value=10,
                key="num_records_patients"
            )

            # Obtener todos los datos para aplicar filtros
            query_patients = pd.read_sql_query(f"SELECT * FROM covid_patients", conn)

            # Aplicar filtros interactivos a los datos de pacientes
            filtered_patients_data = filter_dataframe(query_patients, "patients")

            # Mostrar los datos filtrados limitados por el número seleccionado
            st.dataframe(filtered_patients_data.head(num_records_patients), use_container_width=True)

            # Mostrar estadísticas básicas de los datos filtrados
            with st.expander("📈 Estadísticas de los datos filtrados de pacientes", expanded=True):
                st.write(filtered_patients_data.describe())

    with tab2:
        st.header("Análisis estadístico dinámico")

        # Control para seleccionar número de países en el top
        top_n_countries = st.slider(
            "Cantidad de países ",
            min_value=5,
            max_value=30,
            value=10,
            step=5
        )

        # Visualización de barras con selector de variable a mostrar
        metric_to_show = st.radio(
            "Métrica a visualizar:",
            ["Confirmados", "Muertes", "Ambas"],
            horizontal=True
        )

        # Casos confirmados y muertes por país (con número variable de países)
        aggr_query1 = pd.read_sql_query(f"""
            SELECT country_region, SUM(confirmed) as total_confirmed, SUM(deaths) as total_deaths
            FROM covid_country_stats
            GROUP BY country_region
            ORDER BY total_confirmed DESC
            LIMIT {top_n_countries}
        """, conn)

        col1, col2 = st.columns(2)

        with col1:

            if metric_to_show == "Confirmados":
                fig1 = px.bar(
                    aggr_query1,
                    x='country_region',
                    y='total_confirmed',
                    title=f'Top {top_n_countries} países por casos confirmados',
                    labels={'total_confirmed': 'Casos Confirmados', 'country_region': 'País'},
                    color='total_confirmed',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig1, use_container_width=True)
            elif metric_to_show == "Muertes":
                fig1 = px.bar(
                    aggr_query1,
                    x='country_region',
                    y='total_deaths',
                    title=f'Top {top_n_countries} países por muertes',
                    labels={'total_deaths': 'Muertes', 'country_region': 'País'},
                    color='total_deaths',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig1, use_container_width=True)
            else:
                fig1 = px.bar(
                    aggr_query1,
                    x='country_region',
                    y=['total_confirmed', 'total_deaths'],
                    title=f'Top {top_n_countries} países: Casos confirmados y muertes',
                    labels={'value': 'Cantidad', 'country_region': 'País', 'variable': 'Métrica'},
                    barmode='group'
                )
                st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Tasa de mortalidad por país
            aggr_query3 = pd.read_sql_query(f"""
                SELECT country_region, 
                       SUM(deaths) * 100.0 / NULLIF(SUM(confirmed), 0) as mortality_rate
                FROM covid_country_stats
                GROUP BY country_region
                HAVING SUM(confirmed) > 1000
                ORDER BY mortality_rate DESC
                LIMIT {top_n_countries}
            """, conn)

            fig3 = px.bar(
                aggr_query3,
                x='country_region',
                y='mortality_rate',
                title=f'Top {top_n_countries} países por tasa de mortalidad (países con >1000 casos)',
                labels={'mortality_rate': 'Tasa de Mortalidad (%)', 'country_region': 'País'},
                color='mortality_rate',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Mapa de casos confirmados (utilizando ancho completo)
        st.subheader("🗺️ Mapa global de COVID-19")

        metrics = {
            "Casos confirmados": "confirmed",
            "Fallecimientos": "deaths",
            "Tasa de Incidencia": "incident_rate",
            "Tasa de letalidad": "case_fatality_ratio"
        }

        # Selector de variable para el mapa
        map_metric_selected = st.selectbox(
            "Variable a mostrar en el mapa:",
            list(metrics.keys())
        )

        # Consulta para el mapa basada en la selección
        aggr_query2 = pd.read_sql_query(f"""
            SELECT country_region, SUM(confirmed) as confirmed, 
                SUM(deaths) as deaths, 
                AVG(incident_rate) as incident_rate, 
                AVG(case_fatality_ratio) as case_fatality_ratio,
                AVG(lat) as lat, AVG(long_) as lon
            FROM covid_country_stats
            GROUP BY country_region
        """, conn)
        df_map = pd.DataFrame(aggr_query2)
        map_metric = metrics[map_metric_selected]
        df_map[f'{map_metric}_clean'] = df_map[map_metric].fillna(0)

        # Ajustar etiquetas según la métrica seleccionada
        metric_labels = {
            "confirmed": "Casos Confirmados",
            "deaths": "Muertes",
            "incident_rate": "Tasa de Incidencia",
            "case_fatality_ratio": "Tasa de Mortalidad (%)"
        }
        # Escalas de colores para diferentes métricas
        color_scales = {
            "confirmed": "Reds",
            "deaths": "Purples",
            "incident_rate": "YlOrRd",
            "case_fatality_ratio": "RdBu_r"  # Escala rojo-azul invertida
        }

        fig2 = px.scatter_geo(
            df_map,
            lat='lat',
            lon='lon',
            color=f'{map_metric}_clean',
            size=f'{map_metric}_clean',
            hover_name='country_region',
            title=f'Distribución global de {metric_labels.get(map_metric, map_metric)}',
            projection='natural earth',
            color_continuous_scale=color_scales.get(map_metric, 'Viridis')
        )
        fig2.update_traces(
            marker=dict(
                sizemin=5,
                sizemode='area',
                sizeref=2. * max(df_map[f'{map_metric}_clean']) / 100,
                line=dict(width=0)
            )
        )
        fig2.update_layout(
            height=900,
            geo=dict(
                showland=True,
                landcolor='rgb(217, 217, 217)',  # Color más claro para la tierra
                showocean=True,
                oceancolor='rgb(204, 229, 255)',  # Azul claro para los océanos
                showcoastlines=True,
                coastlinecolor='rgb(150, 150, 150)',
                showlakes=True,
                lakecolor='rgb(204, 229, 255)',
                showcountries=True,
                countrycolor='rgb(100, 100, 100)',
                projection_scale=1,  # Ajuste de zoom inicial
                center=dict(lat=20, lon=0),  # Centrar el mapa
                showframe=False
            ),
            coloraxis_colorbar=dict(
                title=metric_labels.get(map_metric, map_metric),
                thicknessmode="pixels",
                thickness=20,
                lenmode="pixels",
                len=300
            )
        )

        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.header("Análisis estadístico simple")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Distribución por género y edad")

            # Obtener datos de distribución de género
            gender_query = pd.read_sql_query("""
                SELECT sex, COUNT(*) as count
                FROM covid_patients
                GROUP BY sex
                ORDER BY count DESC
            """, conn)
            gender_df = pd.DataFrame(gender_query)
            # Mapear los valores numéricos a etiquetas en español
            gender_df['etiqueta_genero'] = gender_df['sex'].map({
                0: 'Hombre',
                1: 'Mujer'
            })

            fig_gender = px.pie(
                gender_df,
                values='count',
                names='etiqueta_genero',
                title='Distribución por género',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_gender, use_container_width=True)

            # Distribución por grupo de edad
            age_query = pd.read_sql_query("""
                SELECT 
                    CASE 
                        WHEN age < 18 THEN 'menor de 18'
                        WHEN age BETWEEN 18 AND 30 THEN '18-30'
                        WHEN age BETWEEN 31 AND 45 THEN '31-45'
                        WHEN age BETWEEN 46 AND 60 THEN '46-60'
                        WHEN age > 60 THEN 'mayor de 60'
                        ELSE 'no definido'
                    END as age_group,
                    COUNT(*) as count
                FROM covid_patients
                GROUP BY age_group
                ORDER BY 
                    CASE 
                        WHEN age_group = 'menor de 18' THEN 1
                        WHEN age_group = '18-30' THEN 2
                        WHEN age_group = '31-45' THEN 3
                        WHEN age_group = '46-60' THEN 4
                        WHEN age_group = 'mayor de 60' THEN 5
                        ELSE 6
                    END
            """, conn)

            fig_age = px.bar(
                age_query,
                x='age_group',
                y='count',
                title='Distribución por grupo de edad',
                color='count',
                labels={'count': 'Número de pacientes', 'age_group': 'Grupo de edad'},
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_age, use_container_width=True)

        with col2:
            st.subheader("🏥 Condiciones médicas")

            # Consulta de condiciones médicas
            conditions_query = pd.read_sql_query("""
                SELECT 
                    SUM(CASE WHEN pneumonia = 1 THEN 1 ELSE 0 END) as pneumonia,
                    SUM(CASE WHEN diabetes = 1 THEN 1 ELSE 0 END) as diabetes,
                    SUM(CASE WHEN copd = 1 THEN 1 ELSE 0 END) as copd,
                    SUM(CASE WHEN asthma = 1 THEN 1 ELSE 0 END) as asthma,
                    SUM(CASE WHEN cardiovascular = 1 THEN 1 ELSE 0 END) as cardiovascular,
                    SUM(CASE WHEN obesity = 1 THEN 1 ELSE 0 END) as obesity,
                    SUM(CASE WHEN renal_chronic = 1 THEN 1 ELSE 0 END) as renal_chronic,
                    SUM(CASE WHEN tobacco = 1 THEN 1 ELSE 0 END) as tobacco
                FROM covid_patients
            """, conn)

            # Transformar para visualización
            conditions_df = pd.melt(conditions_query)
            conditions_df.columns = ['Condición', 'Cantidad']

            fig_conditions = px.bar(
                conditions_df,
                x='Condición',
                y='Cantidad',
                title='Prevalencia de condiciones médicas',
                color='Cantidad',
                labels={'Cantidad': 'Número de pacientes'},
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_conditions, use_container_width=True)

            # Análisis de comorbilidades y tasas de mortalidad
            mortality_query = pd.read_sql_query("""
                SELECT 
                    CASE WHEN date_died != '9999-99-99' THEN 'Fallecido' ELSE 'Sobreviviente' END as outcome,
                    COUNT(*) as count
                FROM covid_patients
                GROUP BY outcome
            """, conn)

            fig_mortality = px.pie(
                mortality_query,
                values='count',
                names='outcome',
                title='Distribución de desenlaces',
                hole=0.4,
                color_discrete_sequence=['#2ca02c', '#d62728']
            )
            st.plotly_chart(fig_mortality, use_container_width=True)

    # Cerrar la conexión a la base de datos
    conn.close()

    # Pie de página
    st.markdown("---")
    st.subheader("Fuentes de datos")
    st.markdown("""
    #### Datos globales de COVID-19
    - **API:** https://coronavirus.m.pipedream.net/
    - **COVID-19 Data Repository** por el Center for Systems Science and Engineering (CSSE) de Johns Hopkins University
    - [Repositorio en GitHub](https://github.com/CSSEGISandData/COVID-19)
    - Los datos incluyen casos confirmados, muertes, y tasas de incidencia a nivel mundial
    """)
    st.markdown("""
    #### Datos de pacientes con COVID-19
    - **COVID-19 patient's symptoms, status, and medical history**
    - [Dataset en Kaggle](https://www.kaggle.com/datasets/meirnizri/covid19-dataset/data)
    - Los datos incluyen síntomas, estado del paciente, historial médico y factores de riesgo
    """)

    st.caption("""
    **Dashboard desarrollado con Streamlit.** Esta herramienta integra múltiples fuentes de datos para análisis de COVID-19.
    Última actualización: 2025-04-06
    """)

    # Aviso legal
    with st.expander("Aviso legal", expanded=True):
        st.markdown("""
        Este dashboard es solo para fines educativos y de visualización. Los datos pueden no reflejar la información más actualizada sobre COVID-19.

        Todas las fuentes de datos se utilizan de acuerdo con sus respectivas licencias. Las citas completas son:

        1. Dong E, Du H, Gardner L. "An interactive web-based dashboard to track COVID-19 in real time". Lancet Inf Dis. 20(5):533-534. doi: 10.1016/S1473-3099(20)30120-1

        2. Dataset de Kaggle: "COVID-19 patient's symptoms, status, and medical history", proporcionado bajo licencia CC0: Public Domain.
        """)


if __name__ == "__main__":
    main()
