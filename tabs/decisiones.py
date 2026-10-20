import numpy as np
import pandas as pd
import streamlit as st

from tabs.prescriptiva import DIMENSIONES, analizar_segmentos, preparar_datos


# ----------------------------------
# FUNCIONES DE CÁLCULO
# ----------------------------------

def intervalo_de_compra(compradores, clientes, z=1.96):
    """Rango razonable (95%) para el porcentaje real de compra del grupo."""
    if clientes == 0:
        return 0.0, 0.0
    p = compradores / clientes
    centro = (p + z**2 / (2 * clientes)) / (1 + z**2 / clientes)
    margen = z * np.sqrt(p * (1 - p) / clientes + z**2 / (4 * clientes**2)) / (1 + z**2 / clientes)
    return max(centro - margen, 0), min(centro + margen, 1)


def clientes_necesarios_por_grupo(tasa_base, diferencia):
    """Clientes por grupo para distinguir una diferencia de una casualidad (aprox. 80% de seguridad)."""
    if diferencia <= 0:
        return None
    p = (tasa_base + (tasa_base + diferencia)) / 2
    return int(np.ceil(2 * (1.96 + 0.84) ** 2 * p * (1 - p) / diferencia**2))


def filtrar_grupo(datos, caracteristica, grupo):
    return datos[datos[DIMENSIONES[caracteristica]] == grupo]


def mensaje_ejemplo(sub):
    """Mensaje basado solo en datos conocidos: lo que este grupo ya compra."""
    tipo = sub["Tipo_Playera"].mode().iloc[0].lower()
    tela = sub["Tipo_Tela"].mode().iloc[0].lower()
    return (
        f"Hola, gracias por elegir nuestra playera {tipo} de {tela}. "
        "Te invitamos a conocer la Venta Especial de Motos."
    )


# ----------------------------------
# FUNCIÓN PRINCIPAL DE LA PESTAÑA
# ----------------------------------

def mostrar(df):

    st.header("6. Toma de decisiones")

    st.markdown("""
    ### Pregunta de esta etapa

    **¿Cuál es la mejor decisión que podemos tomar hoy para vender más?**

    Reunimos todo lo anterior en una decisión concreta: qué hacer, a quién,
    por qué, quién lo hace y cómo sabremos si funcionó.
    """)

    datos = preparar_datos(df)
    segmentos, tasa_general, margen_general = analizar_segmentos(datos, "compras")

    candidatos = segmentos[segmentos["nivel_compra"] == "alto"].sort_values("fuerza", ascending=False)

    # ----------------------------------
    # 1. DECISIÓN RECOMENDADA
    # ----------------------------------

    st.subheader("1. La decisión recomendada")

    if candidatos.empty:
        st.warning(
            "Ningún grupo compra claramente más que el resto. La mejor decisión es **no segmentar "
            "todavía**: haga una prueba con clientes elegidos al azar y reúna más datos."
        )
        return

    opciones = {f"{f['Característica']}: {f['Grupo']}": i for i, f in candidatos.iterrows()}
    nombres = list(opciones.keys())

    elegido = st.selectbox(
        "Grupo al que se dirigirá la acción (el primero es el mejor respaldado por los datos):",
        nombres,
        key="grupo_decisiones"
    )
    fila = candidatos.loc[opciones[elegido]]
    caracteristica, grupo = fila["Característica"], fila["Grupo"]

    sub = filtrar_grupo(datos, caracteristica, grupo)
    clientes_grupo = len(sub)
    compradores = int(sub["compro_especial"].sum())
    tasa_grupo = compradores / clientes_grupo * 100
    bajo, alto = (x * 100 for x in intervalo_de_compra(compradores, clientes_grupo))
    diferencia = tasa_grupo - tasa_general

    with st.container(border=True):
        st.markdown(
            f"### Lanzar un piloto de la Venta Especial dirigido a clientes de "
            f"**{caracteristica.lower()} = {grupo}**"
        )
        st.markdown(
            f"**Por qué:** en este grupo compró **{tasa_grupo:.0f}%** de los clientes, "
            f"frente a **{tasa_general:.0f}%** en general (**{diferencia:+.0f} puntos**), "
            f"con {clientes_grupo} clientes."
        )
        st.markdown(
            "**Por qué un piloto y no una campaña completa:** la diferencia es clara, pero revisamos "
            f"{len(segmentos)} grupos y algunas diferencias aparecen por casualidad. "
            "Un piloto con grupo de comparación confirma si es real antes de gastar más."
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Compra en el grupo", f"{tasa_grupo:.0f}%")
    c2.metric("Compra general", f"{tasa_general:.0f}%")
    c3.metric("Rango razonable del grupo", f"{bajo:.0f}% a {alto:.0f}%",
              help="Con tan pocos clientes, el porcentaje real podría estar en cualquier punto de este rango.")

    if bajo > tasa_general:
        st.success("Aun en el peor escenario del rango, el grupo compra más que el promedio general.")
    else:
        st.warning(
            "En el peor escenario del rango, el grupo compraría igual o menos que el promedio general. "
            "Es otra razón para comprobarlo con un piloto."
        )

    # ----------------------------------
    # 2. CUÁNTO PODRÍAMOS GANAR
    # ----------------------------------

    st.subheader("2. ¿Cuánto podría ayudar?")

    contactos = st.slider(
        "¿A cuántos clientes de este perfil puede contactar?",
        10, 500, 100, step=10,
        key="contactos_decisiones"
    )

    esperado_grupo = contactos * tasa_grupo / 100
    esperado_general = contactos * tasa_general / 100

    d1, d2, d3 = st.columns(3)
    d1.metric("Compradores esperados si elige este grupo", f"{esperado_grupo:.0f}",
              help=f"Escenario bajo: {contactos * bajo / 100:.0f}. Escenario alto: {contactos * alto / 100:.0f}.")
    d2.metric("Compradores esperados sin segmentar", f"{esperado_general:.0f}")
    d3.metric("Compradores extra", f"{esperado_grupo - esperado_general:+.0f}")

    st.caption(
        f"Escenarios: entre {contactos * bajo / 100:.0f} y {contactos * alto / 100:.0f} compradores "
        "dentro del grupo. Es una estimación a partir de datos históricos, no una promesa."
    )

    # ----------------------------------
    # 3. PLAN DE DECISIÓN
    # ----------------------------------

    st.subheader("3. Plan de decisión")

    responsable = st.text_input(
        "Responsable de la acción:",
        placeholder="Nombre o puesto",
        key="responsable_decisiones"
    )

    necesarios = clientes_necesarios_por_grupo(tasa_general / 100, diferencia / 100)
    mejora_minima = max(diferencia / 2, 1)

    if necesarios:
        if contactos / 2 < necesarios:
            st.warning(
                f"Para confirmar una diferencia de este tamaño se necesitan unos **{necesarios} clientes "
                f"por grupo** (con acción y de comparación). Con {contactos} contactos tendría "
                f"{contactos // 2} por grupo: el resultado sería poco concluyente."
            )
        else:
            st.success(
                f"Con {contactos} contactos tendría {contactos // 2} clientes por grupo; "
                f"se necesitan unos {necesarios} para confirmar la diferencia."
            )

    plan = pd.DataFrame({
        "Elemento": [
            "Indicador a mejorar",
            "Grupo",
            "Evidencia",
            "Acción",
            "Responsable",
            "Siguiente paso",
            "Riesgo",
            "Cómo se medirá el éxito",
        ],
        "Detalle": [
            "Porcentaje de clientes que compra en la Venta Especial de Motos",
            f"{caracteristica}: {grupo} ({clientes_grupo} clientes)",
            f"Compra {tasa_grupo:.0f}% frente a {tasa_general:.0f}% en general",
            "Piloto dirigido a este grupo, con otro grupo similar sin contactar como comparación",
            responsable.strip() or "Por asignar",
            "Dividir al azar a los clientes del grupo en dos mitades y enviar la invitación solo a una",
            "La diferencia podría ser casualidad o deberse a algo que no medimos; el grupo es pequeño",
            f"Ampliar solo si el grupo contactado compra al menos {mejora_minima:.0f} puntos más que el de comparación",
        ]
    })
    st.dataframe(plan, width="stretch", hide_index=True)

    # ----------------------------------
    # 4. MENSAJE DE EJEMPLO
    # ----------------------------------

    st.subheader("4. Mensaje de ejemplo")

    st.caption(
        "Usa solo lo que sabemos de este grupo: el tipo y tela de playera que más compra. "
        "No promete descuentos ni beneficios que no estén definidos. Edítelo libremente."
    )

    mensaje = st.text_area(
        "Mensaje:",
        value=mensaje_ejemplo(sub),
        key=f"mensaje_decisiones_{caracteristica}_{grupo}",
        height=100
    )

    # ----------------------------------
    # 5. SEGUNDA DECISIÓN: MARGEN
    # ----------------------------------

    st.subheader("5. Decisión complementaria: cuidar el margen")

    margen_tipos = datos.groupby("Tipo_Playera")["Margen_MXN"].mean().sort_values(ascending=False)
    mejor_tipo, peor_tipo = margen_tipos.index[0], margen_tipos.index[-1]

    st.markdown(
        f"- **Dar más visibilidad a la playera {mejor_tipo}:** deja ${margen_tipos.iloc[0]:,.0f} por playera, "
        f"contra ${margen_general:,.0f} en promedio.\n"
        f"- **Revisar precio o costo de la playera {peor_tipo}** antes de promoverla: deja solo "
        f"${margen_tipos.iloc[-1]:,.0f} por playera."
    )
    st.caption(
        "El margen es de la playera vendida, no de la Venta Especial de Motos, "
        "así que no se suman: son dos decisiones separadas."
    )

    # ----------------------------------
    # 6. QUÉ NO HACER
    # ----------------------------------

    st.subheader("6. Lo que NO recomendamos")

    por_evitar = segmentos[segmentos["nivel_compra"] == "bajo"]
    lineas = [
        "No usar el modelo de la pestaña Predictiva para elegir clientes: no supera al azar.",
        "No ofrecer descuentos a grupos que ya compran más: no hay evidencia de que los necesiten.",
        "No ampliar a toda la base hasta confirmar el resultado del piloto.",
    ]
    for _, f in por_evitar.iterrows():
        lineas.insert(0, f"No aumentar la inversión en **{f['Característica'].lower()}: {f['Grupo']}** "
                         f"(compra {f['Compra']:.0f}%) hasta probar un mensaje distinto.")
    for linea in lineas:
        st.markdown(f"- {linea}")

    # ----------------------------------
    # DESCARGA
    # ----------------------------------

    resumen = pd.concat([
        plan,
        pd.DataFrame({"Elemento": ["Mensaje"], "Detalle": [mensaje]})
    ], ignore_index=True)

    st.download_button(
        "Descargar plan de decisión (CSV)",
        data=resumen.to_csv(index=False).encode("utf-8-sig"),
        file_name="plan_de_decision.csv",
        mime="text/csv",
        key="descarga_decisiones"
    )

    st.caption(
        "Datos ficticios con fines de aprendizaje. Esta es una recomendación de apoyo: "
        "la decisión final la toma una persona que conoce el negocio."
    )
