import streamlit as st
import pandas as pd


COLUMNA_OBJETIVO = "Compró Venta Especial Motos"


# ----------------------------------
# FUNCIONES DE DIAGNÓSTICO
# ----------------------------------

def tabla_faltantes(df):
    faltantes = df.isnull().sum()
    return pd.DataFrame({
        "Variable": df.columns,
        "Valores faltantes": faltantes.values,
        "% faltante": (faltantes.values / len(df) * 100).round(1)
    })


def contar_duplicados(df):
    exactos = int(df.duplicated().sum())
    por_id = int(df["ID_Cliente"].duplicated().sum()) if "ID_Cliente" in df.columns else 0
    return exactos, por_id


def tabla_reglas_validez(df):
    """Cada regla devuelve la máscara de registros que la incumplen."""
    reglas = {
        "Edad entre 15 y 100 años": ~df["Edad"].between(15, 100),
        "Precio mayor a 0": df["Precio_MXN"] <= 0,
        "Costo mayor o igual a 0": df["Costo_MXN"] < 0,
        "Costo menor al precio": df["Costo_MXN"] >= df["Precio_MXN"],
        "Margen = Precio - Costo": df["Margen_MXN"] != (df["Precio_MXN"] - df["Costo_MXN"]),
        "Año de venta entre 2000 y 2026": ~df["Año_Venta"].between(2000, 2026),
        "Respuesta de campaña es Sí o No": ~df[COLUMNA_OBJETIVO].isin(["Sí", "No"]),
    }
    filas = []
    for regla, incumple in reglas.items():
        filas.append({
            "Regla": regla,
            "Registros que la incumplen": int(incumple.sum()),
            "Resultado": "✅ Cumple" if incumple.sum() == 0 else "⚠️ Revisar"
        })
    return pd.DataFrame(filas), reglas


def detectar_atipicos_iqr(serie):
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    limite_inf, limite_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (serie < limite_inf) | (serie > limite_sup), limite_inf, limite_sup


def tabla_espacios_y_mayusculas(df, columnas_texto):
    """Detecta categorías que solo difieren por espacios o mayúsculas."""
    filas = []
    for col in columnas_texto:
        originales = df[col].nunique()
        normalizadas = df[col].str.strip().str.lower().nunique()
        con_espacios = int((df[col] != df[col].str.strip()).sum())
        filas.append({
            "Variable": col,
            "Categorías originales": originales,
            "Categorías tras normalizar": normalizadas,
            "Registros con espacios sobrantes": con_espacios,
            "Resultado": "✅ Consistente" if originales == normalizadas and con_espacios == 0 else "⚠️ Revisar"
        })
    return pd.DataFrame(filas)


# ----------------------------------
# FUNCIÓN PRINCIPAL DE LA PESTAÑA
# ----------------------------------

def mostrar(df):
    """Muestra la pestaña 2 y devuelve df_limpio.
    El dataframe original (df) nunca se modifica."""

    st.header("2. Limpiar los datos")

    st.markdown("""
    ### Pregunta de esta etapa

    **¿Los datos son confiables para analizarlos y qué debemos corregir, con qué razón?**

    Antes de cambiar algo, primero **revisamos**. Un valor extraño no siempre es un error,
    y cada cambio debe quedar justificado.
    """)

    columnas_texto = df.select_dtypes(include=["object", "string"]).columns.tolist()
    columnas_numericas = [
        c for c in df.select_dtypes(include="number").columns
    ]

    # ----------------------------------
    # 1. FALTANTES
    # ----------------------------------

    st.subheader("1. Valores faltantes")

    faltantes = tabla_faltantes(df)
    total_faltantes = int(faltantes["Valores faltantes"].sum())

    if total_faltantes == 0:
        st.success("No hay valores faltantes en ninguna variable.")
    else:
        st.warning(f"Hay {total_faltantes} valores faltantes. "
                   "Antes de imputar, pregunta qué significa que falten.")
        st.dataframe(
            faltantes[faltantes["Valores faltantes"] > 0],
            use_container_width=True
        )

    # ----------------------------------
    # 2. DUPLICADOS
    # ----------------------------------

    st.subheader("2. Duplicados")

    duplicados_exactos, duplicados_id = contar_duplicados(df)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Filas duplicadas exactas", duplicados_exactos)
    with col2:
        st.metric("ID_Cliente repetidos", duplicados_id)

    st.caption(
        "Un ID repetido no siempre es duplicado: un cliente podría comprar más de una vez. "
        "Se debe verificar el significado de cada fila antes de eliminarla."
    )

    # ----------------------------------
    # 3. REGLAS DE VALIDEZ
    # ----------------------------------

    st.subheader("3. Reglas de validez")

    resumen_reglas, mascaras_reglas = tabla_reglas_validez(df)
    st.dataframe(resumen_reglas, use_container_width=True, hide_index=True)

    regla_elegida = st.selectbox(
        "Ver registros que incumplen una regla:",
        list(mascaras_reglas.keys()),
        key="regla_limpiar"
    )
    incumplen = df[mascaras_reglas[regla_elegida]]
    if incumplen.empty:
        st.info("Ningún registro incumple esta regla.")
    else:
        st.dataframe(incumplen, use_container_width=True)

    # ----------------------------------
    # 4. CONSISTENCIA DE CATEGORÍAS
    # ----------------------------------

    st.subheader("4. Consistencia de categorías")

    st.dataframe(
        tabla_espacios_y_mayusculas(df, columnas_texto),
        use_container_width=True,
        hide_index=True
    )

    # ----------------------------------
    # 5. VALORES ATÍPICOS
    # ----------------------------------

    st.subheader("5. Valores atípicos")

    st.markdown(
        "Se marcan con la regla del rango intercuartílico (IQR). "
        "**Marcar no es eliminar**: un valor extremo puede ser real e importante."
    )

    filas_atipicos = []
    for col in columnas_numericas:
        es_atipico, inf, sup = detectar_atipicos_iqr(df[col])
        filas_atipicos.append({
            "Variable": col,
            "Límite inferior": round(inf, 1),
            "Límite superior": round(sup, 1),
            "Atípicos": int(es_atipico.sum())
        })
    st.dataframe(pd.DataFrame(filas_atipicos), use_container_width=True, hide_index=True)

    variable_atipicos = st.selectbox(
        "Ver atípicos de la variable:",
        columnas_numericas,
        key="variable_atipicos_limpiar"
    )
    es_atipico, _, _ = detectar_atipicos_iqr(df[variable_atipicos])
    if es_atipico.sum() == 0:
        st.info("No hay valores atípicos en esta variable.")
    else:
        st.dataframe(df[es_atipico], use_container_width=True)

    # ----------------------------------
    # 6. DECISIONES DE LIMPIEZA
    # ----------------------------------

    st.subheader("6. Decisiones de limpieza")

    st.markdown(
        "Elige qué tratamientos aplicar. Cada uno crea una copia limpia; "
        "los datos originales quedan intactos."
    )

    quitar_espacios = st.checkbox(
        "Quitar espacios sobrantes en variables de texto",
        value=True,
        key="quitar_espacios_limpiar"
    )
    quitar_duplicados = st.checkbox(
        "Eliminar filas duplicadas exactas",
        value=duplicados_exactos > 0,
        key="quitar_duplicados_limpiar"
    )
    crear_binaria = st.checkbox(
        f"Crear columna 0/1 a partir de '{COLUMNA_OBJETIVO}' (Sí = 1, No = 0)",
        value=True,
        key="crear_binaria_limpiar"
    )

    razon = st.text_area(
        "Escribe la razón de tus decisiones:",
        placeholder="Ej.: No hay faltantes ni duplicados; solo se codifica la respuesta para poder calcular tasas.",
        key="razon_limpiar"
    )

    # ----------------------------------
    # CONSTRUCCIÓN DEL DATAFRAME LIMPIO
    # ----------------------------------

    df_limpio = df.copy()
    bitacora = []

    if quitar_espacios:
        for col in columnas_texto:
            df_limpio[col] = df_limpio[col].str.strip()
        bitacora.append("Se quitaron espacios sobrantes en variables de texto.")

    if quitar_duplicados:
        antes = len(df_limpio)
        df_limpio = df_limpio.drop_duplicates()
        bitacora.append(f"Se eliminaron {antes - len(df_limpio)} filas duplicadas exactas.")

    if crear_binaria:
        df_limpio["compro_venta_especial"] = (
            df_limpio[COLUMNA_OBJETIVO].map({"Sí": 1, "No": 0})
        )
        bitacora.append("Se creó 'compro_venta_especial' (1 = Sí, 0 = No).")

    # ----------------------------------
    # RESULTADO
    # ----------------------------------

    st.subheader("Resultado")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filas originales", len(df))
    with col2:
        st.metric("Filas tras limpiar", len(df_limpio))
    with col3:
        st.metric("Columnas tras limpiar", len(df_limpio.columns))

    if bitacora:
        st.markdown("**Bitácora de cambios**")
        for linea in bitacora:
            st.markdown(f"- {linea}")
    else:
        st.info("No se aplicó ningún tratamiento: los datos limpios son iguales a los originales.")

    if not razon.strip():
        st.warning("Falta escribir la razón de tus decisiones.")

    return df_limpio
