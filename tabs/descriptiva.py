import streamlit as st
import pandas as pd


COLUMNA_OBJETIVO = "Compró Venta Especial Motos"

# Indicador -> (columna calculada, formato)
INDICADORES = {
    "Número de clientes": ("clientes", "{:,.0f}"),
    "Ingresos totales (MXN)": ("ingresos", "${:,.0f}"),
    "Costo total (MXN)": ("costo", "${:,.0f}"),
    "Margen total (MXN)": ("margen", "${:,.0f}"),
    "Margen sobre el precio (%)": ("margen_pct", "{:.1f}%"),
    "Precio promedio (MXN)": ("precio_prom", "${:,.0f}"),
    "Costo promedio (MXN)": ("costo_prom", "${:,.0f}"),
    "Edad promedio": ("edad_prom", "{:.1f}"),
    "Compró la Venta Especial (%)": ("tasa_compra", "{:.1f}%"),
}


# ----------------------------------
# FUNCIONES DE CÁLCULO
# ----------------------------------

def preparar_datos(df):
    datos = df.copy()
    datos["compro_especial"] = (datos[COLUMNA_OBJETIVO].astype(str).str.strip() == "Sí").astype(int)
    return datos


def calcular_indicadores(datos):
    """Indicadores de un conjunto de clientes. Cada fila es un cliente y una venta."""
    clientes = len(datos)
    ingresos = datos["Precio_MXN"].sum()
    margen = datos["Margen_MXN"].sum()
    return pd.Series({
        "clientes": clientes,
        "ingresos": ingresos,
        "costo": datos["Costo_MXN"].sum(),
        "margen": margen,
        # Se evita dividir entre cero si el grupo queda vacío
        "margen_pct": margen / ingresos * 100 if ingresos else 0,
        "precio_prom": datos["Precio_MXN"].mean() if clientes else 0,
        "costo_prom": datos["Costo_MXN"].mean() if clientes else 0,
        "edad_prom": datos["Edad"].mean() if clientes else 0,
        "tasa_compra": datos["compro_especial"].mean() * 100 if clientes else 0,
    })


def indicadores_por_grupo(datos, columna):
    return datos.groupby(columna).apply(calcular_indicadores, include_groups=False)


def tabla_amigable(tabla, nombre_grupo):
    """Renombra columnas a lenguaje de negocio."""
    nombres = {col: nombre for nombre, (col, _) in INDICADORES.items()}
    tabla = tabla.rename(columns=nombres)
    tabla.index.name = nombre_grupo
    return tabla


def formato_columnas(tabla):
    return {
        nombre: formato for nombre, (_, formato) in INDICADORES.items()
        if nombre in tabla.columns
    }


# ----------------------------------
# FUNCIÓN PRINCIPAL DE LA PESTAÑA
# ----------------------------------

def mostrar(df):

    st.header("3. Estadística descriptiva")

    st.markdown("""
    ### Pregunta de esta etapa

    **¿Cómo nos está yendo y qué diferencias hay entre nuestros clientes?**

    Aquí solo **describimos** lo que pasó. Una diferencia entre grupos no
    explica por qué ocurre ni prueba que una estrategia vaya a funcionar.
    """)

    datos = preparar_datos(df)

    # ----------------------------------
    # 1. INDICADORES GENERALES
    # ----------------------------------

    st.subheader("1. Indicadores generales")

    general = calcular_indicadores(datos)

    fila1 = st.columns(4)
    fila1[0].metric("Clientes", f"{general['clientes']:,.0f}")
    fila1[1].metric("Ingresos totales", f"${general['ingresos']:,.0f}")
    fila1[2].metric("Costo total", f"${general['costo']:,.0f}")
    fila1[3].metric("Margen total", f"${general['margen']:,.0f}")

    fila2 = st.columns(4)
    fila2[0].metric("Margen sobre el precio", f"{general['margen_pct']:.1f}%",
                    help="De cada $100 que cobramos, cuánto nos queda después del costo.")
    fila2[1].metric("Precio promedio", f"${general['precio_prom']:,.0f}")
    fila2[2].metric("Edad promedio", f"{general['edad_prom']:.1f} años")
    fila2[3].metric("Compró la Venta Especial", f"{general['tasa_compra']:.1f}%",
                    help="Porcentaje del total de clientes que respondió «Sí».")

    st.caption(
        "Cada fila de los datos es un cliente con una venta, por eso "
        "«clientes» y «ventas» coinciden. Costo y margen son por playera, en pesos."
    )

    # ----------------------------------
    # 2. POR TIPO DE PLAYERA
    # ----------------------------------

    st.subheader("2. Resultados por tipo de playera")

    por_tipo = indicadores_por_grupo(datos, "Tipo_Playera")

    indicador_tipo = st.selectbox(
        "Indicador a comparar:",
        list(INDICADORES.keys()),
        index=3,
        key="indicador_tipo_descriptiva"
    )
    columna_tipo = INDICADORES[indicador_tipo][0]

    st.bar_chart(
        por_tipo[columna_tipo].sort_values(ascending=False),
        horizontal=True
    )

    tabla_tipo = tabla_amigable(por_tipo, "Tipo de playera")
    st.dataframe(
        tabla_tipo.style.format(formato_columnas(tabla_tipo)),
        width="stretch"
    )

    # ----------------------------------
    # 3. COMPARACIÓN POR GÉNERO
    # ----------------------------------

    st.subheader("3. Comparación por género")

    por_genero = indicadores_por_grupo(datos, "Genero")

    st.markdown("**Todos los indicadores, lado a lado**")
    tabla_genero = tabla_amigable(por_genero, "Género")
    st.dataframe(
        tabla_genero.style.format(formato_columnas(tabla_genero)),
        width="stretch"
    )

    indicador_genero = st.selectbox(
        "Indicador a comparar entre géneros:",
        list(INDICADORES.keys()),
        index=4,
        key="indicador_genero_descriptiva"
    )
    columna_genero, formato_genero = INDICADORES[indicador_genero]

    valores = por_genero[columna_genero]
    st.bar_chart(valores)

    # Lectura en lenguaje claro
    mayor, menor = valores.idxmax(), valores.idxmin()
    if valores.max() > 0:
        diferencia = (valores.max() - valores.min()) / valores.max() * 100
        st.markdown(
            f"**Lectura:** el valor más alto en *{indicador_genero}* es de **{mayor}** "
            f"({formato_genero.format(valores.max())}) y el más bajo de **{menor}** "
            f"({formato_genero.format(valores.min())}). La diferencia es de {diferencia:.0f}%."
        )
        if diferencia < 10:
            st.info(
                "La diferencia es pequeña. Con grupos de este tamaño podría deberse "
                "al azar, así que no conviene tomar decisiones solo con ella."
            )

    # Tipo de playera dentro de cada género
    st.markdown("**¿Qué tipo de playera compra cada género?**")

    st.caption("Porcentaje de las compras de cada género (cada columna suma 100%).")
    mezcla = (
        pd.crosstab(datos["Tipo_Playera"], datos["Genero"], normalize="columns") * 100
    ).round(1)
    mezcla.index.name = "Tipo de playera"
    st.dataframe(mezcla.style.format("{:.1f}%"), width="stretch")

    # Margen por tipo y género
    st.markdown("**Margen promedio por playera: tipo y género**")
    margen_cruzado = datos.pivot_table(
        index="Tipo_Playera", columns="Genero", values="Margen_MXN", aggfunc="mean"
    )
    margen_cruzado.index.name = "Tipo de playera"
    st.dataframe(margen_cruzado.style.format("${:,.0f}"), width="stretch")

    st.caption(
        "Cada casilla agrupa pocos clientes. Revise la columna «Número de clientes» "
        "antes de confiar en una diferencia."
    )

    # ----------------------------------
    # 4. TENDENCIA POR AÑO
    # ----------------------------------

    st.subheader("4. Evolución por año")

    por_anio = indicadores_por_grupo(datos, "Año_Venta")
    indicador_anio = st.selectbox(
        "Indicador a ver por año:",
        list(INDICADORES.keys()),
        index=1,
        key="indicador_anio_descriptiva"
    )
    st.line_chart(por_anio[INDICADORES[indicador_anio][0]])
    st.caption(
        "El número de ventas cambia cada año, así que un total más alto puede significar "
        "simplemente más ventas, no mejores resultados por venta."
    )
