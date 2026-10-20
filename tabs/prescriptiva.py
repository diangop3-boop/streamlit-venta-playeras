import numpy as np
import pandas as pd
import streamlit as st


COLUMNA_OBJETIVO = "Compró Venta Especial Motos"
MINIMO_CLIENTES = 30     # menos clientes que esto: no se saca conclusión
UMBRAL_CLARO = 2.0       # diferencia suficientemente grande para no ser casualidad típica

OBJETIVOS = {
    "Que más clientes compren en la Venta Especial": "compras",
    "Que cada venta nos deje más margen": "margen",
    "Ambos": "ambos",
}

# Nombre amigable -> columna
DIMENSIONES = {
    "Género": "Genero",
    "Rango de edad": "rango_edad",
    "Tipo de playera": "Tipo_Playera",
    "Talla": "Talla",
    "Color": "Color",
    "Tela": "Tipo_Tela",
}


# ----------------------------------
# FUNCIONES DE CÁLCULO
# ----------------------------------

def preparar_datos(df):
    datos = df.copy()
    datos["compro_especial"] = (datos[COLUMNA_OBJETIVO].astype(str).str.strip() == "Sí").astype(int)
    datos["rango_edad"] = pd.cut(
        datos["Edad"], bins=[0, 29, 35, 40, 200],
        labels=["29 años o menos", "30 a 35 años", "36 a 40 años", "Más de 40 años"]
    ).astype(str)
    return datos


def nivel_de_senal(z, clientes):
    """Alto / bajo / normal, solo si hay suficientes clientes y la diferencia es clara."""
    if clientes < MINIMO_CLIENTES:
        return "normal"
    if z >= UMBRAL_CLARO:
        return "alto"
    if z <= -UMBRAL_CLARO:
        return "bajo"
    return "normal"


def decidir_accion(nivel_compra, nivel_margen, objetivo):
    """Reglas transparentes: (qué hacer, prioridad)."""
    mira_compras = objetivo in ("compras", "ambos")
    mira_margen = objetivo in ("margen", "ambos")

    if mira_compras and nivel_compra == "alto" and nivel_margen != "bajo":
        return "Priorizar: dirigir la próxima campaña a este grupo.", 3
    if mira_compras and nivel_compra == "alto" and nivel_margen == "bajo":
        return "Aprovechar su interés cuidando el margen: evitar ofrecerle descuentos.", 2
    if mira_margen and nivel_margen == "alto":
        return "Darle más visibilidad: es donde cada venta deja más.", 3 if nivel_compra != "bajo" else 2
    if mira_compras and nivel_compra == "bajo":
        return "No ampliar la inversión todavía: probar un mensaje distinto y medir.", 1
    if mira_margen and nivel_margen == "bajo":
        return "Revisar precio o costo de este grupo antes de promoverlo.", 1
    return "Sin acción especial: no hay diferencia clara.", 0


def analizar_segmentos(datos, objetivo):
    tasa_general = datos["compro_especial"].mean()
    margen_general = datos["Margen_MXN"].mean()
    desviacion_margen = datos["Margen_MXN"].std()

    filas = []
    for nombre_dim, columna in DIMENSIONES.items():
        for grupo, sub in datos.groupby(columna):
            clientes = len(sub)
            tasa = sub["compro_especial"].mean()
            margen = sub["Margen_MXN"].mean()

            z_compra = (tasa - tasa_general) / np.sqrt(tasa_general * (1 - tasa_general) / clientes)
            z_margen = (margen - margen_general) / (desviacion_margen / np.sqrt(clientes))

            nivel_compra = nivel_de_senal(z_compra, clientes)
            nivel_margen = nivel_de_senal(z_margen, clientes)
            accion, prioridad = decidir_accion(nivel_compra, nivel_margen, objetivo)

            fuerza = {"compras": abs(z_compra), "margen": abs(z_margen),
                      "ambos": max(abs(z_compra), abs(z_margen))}[objetivo]

            filas.append({
                "Característica": nombre_dim,
                "Grupo": grupo,
                "Clientes": clientes,
                "Compra": tasa * 100,
                "Margen promedio": margen,
                "nivel_compra": nivel_compra,
                "nivel_margen": nivel_margen,
                "Acción recomendada": accion,
                "prioridad": prioridad,
                "fuerza": fuerza,
            })
    return pd.DataFrame(filas), tasa_general * 100, margen_general


def texto_evidencia(fila, tasa_general, margen_general):
    partes = []
    if fila["nivel_compra"] != "normal":
        partes.append(f"compra {fila['Compra']:.0f}% (general: {tasa_general:.0f}%)")
    if fila["nivel_margen"] != "normal":
        partes.append(f"margen ${fila['Margen promedio']:,.0f} por playera "
                      f"(general: ${margen_general:,.0f})")
    if not partes:
        partes.append(f"compra {fila['Compra']:.0f}%, margen ${fila['Margen promedio']:,.0f}")
    return f"{' y '.join(partes).capitalize()}, con {fila['Clientes']} clientes."


# ----------------------------------
# FUNCIÓN PRINCIPAL DE LA PESTAÑA
# ----------------------------------

def mostrar(df):

    st.header("5. Estadística prescriptiva")

    st.markdown("""
    ### Pregunta de esta etapa

    **¿Qué acción de marketing conviene tomar con cada tipo de cliente?**

    Aquí convertimos lo que observamos en **reglas claras** del tipo
    «si el cliente es así, entonces hacemos esto». Cada recomendación muestra
    la evidencia que la respalda.
    """)

    st.info(
        "En la pestaña Predictiva vimos que el modelo no es confiable con estos datos. "
        "Por eso estas recomendaciones **no usan predicciones**: se basan solo en lo que "
        "realmente ocurrió con nuestros clientes."
    )

    datos = preparar_datos(df)

    # ----------------------------------
    # PASO 1. OBJETIVO
    # ----------------------------------

    st.subheader("Paso 1. ¿Qué quiere lograr?")

    objetivo_texto = st.radio(
        "Objetivo principal:",
        list(OBJETIVOS.keys()),
        key="objetivo_prescriptiva"
    )
    objetivo = OBJETIVOS[objetivo_texto]

    segmentos, tasa_general, margen_general = analizar_segmentos(datos, objetivo)

    # ----------------------------------
    # PASO 2. PLAN DE ACCIÓN
    # ----------------------------------

    st.subheader("Paso 2. Acciones recomendadas")

    con_accion = segmentos[segmentos["prioridad"] > 0].sort_values(
        ["prioridad", "fuerza"], ascending=False
    )

    if con_accion.empty:
        st.warning(
            "Ningún grupo se distingue claramente de los demás para este objetivo. "
            "La recomendación es no segmentar todavía y reunir más datos."
        )
    else:
        maximo = min(3, len(con_accion))
        if len(con_accion) > 1:
            maximo = st.slider(
                "¿Cuántas recomendaciones mostrar?", 1, min(8, len(con_accion)),
                maximo, key="cantidad_prescriptiva"
            )
        for _, fila in con_accion.head(maximo).iterrows():
            with st.container(border=True):
                st.markdown(f"**Si el cliente es: {fila['Característica']} → {fila['Grupo']}**")
                st.markdown(f"➡️ **Entonces:** {fila['Acción recomendada']}")
                st.caption(f"Evidencia: {texto_evidencia(fila, tasa_general, margen_general)}")

    # ----------------------------------
    # PASO 3. TODOS LOS GRUPOS
    # ----------------------------------

    st.subheader("Paso 3. Revise todos los grupos")

    caracteristica = st.selectbox(
        "Característica a revisar:",
        ["Todas"] + list(DIMENSIONES.keys()),
        key="caracteristica_prescriptiva"
    )

    tabla = segmentos if caracteristica == "Todas" else segmentos[segmentos["Característica"] == caracteristica]
    tabla = tabla.assign(**{
        "¿Diferencia clara?": np.where(
            tabla["Clientes"] < MINIMO_CLIENTES, "Pocos clientes",
            np.where((tabla["nivel_compra"] != "normal") | (tabla["nivel_margen"] != "normal"),
                     "Sí", "No")
        )
    })[["Característica", "Grupo", "Clientes", "Compra", "Margen promedio",
        "¿Diferencia clara?", "Acción recomendada"]]

    st.dataframe(
        tabla.style.format({"Compra": "{:.0f}%", "Margen promedio": "${:,.0f}"}),
        width="stretch",
        hide_index=True
    )

    st.caption(
        f"Referencia general: {tasa_general:.0f}% de los clientes compró y el margen "
        f"promedio es ${margen_general:,.0f} por playera."
    )

    # ----------------------------------
    # PASO 4. CUIDADOS Y CÓMO COMPROBAR
    # ----------------------------------

    st.subheader("Paso 4. Antes de invertir, compruebe")

    revisados = len(segmentos)
    por_azar = revisados * 0.046

    st.warning(
        f"Revisamos {revisados} grupos. Aun si no existiera ninguna diferencia real, "
        f"es normal que **{por_azar:.0f} o {por_azar + 1:.0f} grupos parezcan destacar solo por casualidad**. "
        "Por eso una recomendación es una *hipótesis* que hay que comprobar, no una certeza."
    )

    with st.expander("Cómo comprobarla con una prueba pequeña", expanded=True):
        st.markdown("""
        1. Elija **un** grupo recomendado.
        2. Divida a sus clientes al azar en dos: uno recibe la acción y el otro **no** (grupo de comparación).
        3. Mida en ambos el mismo indicador: el porcentaje que compra y el margen por venta.
        4. Si el grupo con acción no mejora claramente, no amplíe la inversión.
        """)

    with st.expander("Límites de estas recomendaciones"):
        st.markdown("""
        - **Solo vemos asociaciones:** que un grupo compre más no significa que ser de ese grupo sea la causa.
        - **Pocos clientes:** con 250 registros, los grupos pequeños dan resultados inestables.
        - **No conocemos el contexto:** no sabemos cuánto costó contactar a cada cliente ni qué más ocurrió en la campaña.
        - **Datos ficticios:** el ejercicio es para aprender, no para decidir inversión real.
        - **Una persona decide:** estas reglas son apoyo, la decisión final es de quien conoce el negocio.
        """)
