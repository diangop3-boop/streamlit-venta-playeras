import streamlit as st
import pandas as pd

from tabs import entender, limpiar, descriptiva, predictiva, prescriptiva, decisiones


# ----------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------

st.set_page_config(
    page_title="Sales Analytics",
    page_icon="📊",
    layout="wide"
)


# ----------------------------------
# TÍTULO DE LA APLICACIÓN
# ----------------------------------

st.title("📊 Sales Analytics")

st.markdown("""
Aplicación para analizar información de ventas
y convertir datos en decisiones de ventas.
""")


# ----------------------------------
# CARGAR DATASET
# ----------------------------------

df = pd.read_csv(
    "data/ventas_playeras_ficticias-3.csv"
)


# ----------------------------------
# CREAR PESTAÑAS
# ----------------------------------

tab_entender, tab_limpiar, tab_descriptiva, tab_predictiva, tab_prescriptiva, tab_decisiones = st.tabs([
    "1. Entender",
    "2. Limpiar",
    "3. Descriptiva",
    "4. Predictiva",
    "5. Prescriptiva",
    "6. Decisiones"
])


# ----------------------------------
# 1. ENTENDER LOS DATOS
# ----------------------------------

with tab_entender:
    entender.mostrar(df)


# ----------------------------------
# 2. LIMPIAR LOS DATOS
# ----------------------------------

with tab_limpiar:
    df_limpio = limpiar.mostrar(df)


# ----------------------------------
# 3 A 6. ANÁLISIS Y DECISIONES
# ----------------------------------

with tab_descriptiva:
    descriptiva.mostrar(df_limpio)


with tab_predictiva:
    predictiva.mostrar(df_limpio)


with tab_prescriptiva:
    prescriptiva.mostrar(df_limpio)


with tab_decisiones:
    decisiones.mostrar(df_limpio)
