import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


COLUMNA_OBJETIVO = "Compró Venta Especial Motos"

# Nombre de la columna -> (nombre amigable, tipo)
CARACTERISTICAS = {
    "Genero": ("Género", "texto"),
    "Edad": ("Edad", "numero"),
    "Tipo_Playera": ("Tipo de playera", "texto"),
    "Talla": ("Talla", "texto"),
    "Color": ("Color", "texto"),
    "Tipo_Tela": ("Tela", "texto"),
    "Precio_MXN": ("Precio de la playera", "numero"),
    "Año_Venta": ("Año de la venta", "numero"),
}


# ----------------------------------
# FUNCIONES DEL MODELO
# ----------------------------------

def crear_modelo(columnas):
    """Bosque de árboles: muchas reglas sencillas que 'votan' juntas."""
    columnas_texto = [c for c in columnas if CARACTERISTICAS[c][1] == "texto"]
    preparacion = ColumnTransformer(
        [("texto", OneHotEncoder(handle_unknown="ignore"), columnas_texto)],
        remainder="passthrough"
    )
    return Pipeline([
        ("preparacion", preparacion),
        ("modelo", RandomForestClassifier(
            n_estimators=200, max_depth=4, random_state=42
        )),
    ])


@st.cache_data(show_spinner=False)
def evaluar_modelo(datos, columnas):
    """Prueba el modelo con clientes que NO vio al aprender (validación cruzada)."""
    columnas = list(columnas)
    X = datos[columnas]
    y = (datos[COLUMNA_OBJETIVO].str.strip() == "Sí").astype(int)

    validacion = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probabilidades = cross_val_predict(
        crear_modelo(columnas), X, y, cv=validacion, method="predict_proba"
    )[:, 1]
    predicciones = (probabilidades >= 0.5).astype(int)

    # Qué tanto pesa cada dato: se mide con clientes de prueba apartados
    X_ent, X_prueba, y_ent, y_prueba = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    modelo_prueba = crear_modelo(columnas).fit(X_ent, y_ent)
    importancia = permutation_importance(
        modelo_prueba, X_prueba, y_prueba, n_repeats=20, random_state=42
    ).importances_mean

    tn, fp, fn, tp = confusion_matrix(y, predicciones).ravel()
    return {
        "aciertos": accuracy_score(y, predicciones),
        "adivinando": max(y.mean(), 1 - y.mean()),
        "separacion": roc_auc_score(y, probabilidades),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "importancia": dict(zip(columnas, importancia)),
        "total": len(y),
    }


@st.cache_resource(show_spinner=False)
def entrenar_modelo_final(datos, columnas):
    columnas = list(columnas)
    y = (datos[COLUMNA_OBJETIVO].str.strip() == "Sí").astype(int)
    return crear_modelo(columnas).fit(datos[columnas], y)


def veredicto(separacion):
    """Traduce la calidad del modelo a una conclusión de negocio."""
    if separacion < 0.60:
        return "no_sirve"
    if separacion < 0.70:
        return "poco"
    return "util"


# ----------------------------------
# FUNCIÓN PRINCIPAL DE LA PESTAÑA
# ----------------------------------

def mostrar(df):

    st.header("4. Estadística predictiva")

    st.markdown("""
    ### Pregunta de esta etapa

    **¿Podemos adivinar qué clientes tienen más probabilidad de comprar en la Venta Especial de Motos?**

    La idea: el programa estudia a los clientes que ya conocemos, aprende patrones
    y trata de reconocerlos en clientes nuevos. **Es una ayuda para decidir, no una garantía.**
    """)

    datos = df.copy()
    datos[COLUMNA_OBJETIVO] = datos[COLUMNA_OBJETIVO].astype(str).str.strip()

    # ----------------------------------
    # PASO 1. ELEGIR QUÉ DATOS USAR
    # ----------------------------------

    st.subheader("Paso 1. ¿Qué datos del cliente quiere considerar?")

    nombres_a_columnas = {nombre: col for col, (nombre, _) in CARACTERISTICAS.items()}
    elegidos = st.multiselect(
        "Datos que el programa puede mirar:",
        list(nombres_a_columnas.keys()),
        default=list(nombres_a_columnas.keys()),
        key="datos_predictiva"
    )

    st.caption(
        "No se incluyen el identificador del cliente, el costo ni el margen: "
        "no describen al cliente, son datos internos de la empresa."
    )

    if len(elegidos) == 0:
        st.warning("Elija al menos un dato para continuar.")
        return

    columnas = tuple(nombres_a_columnas[n] for n in elegidos)

    # ----------------------------------
    # PASO 2. ¿QUÉ TAN BIEN FUNCIONA?
    # ----------------------------------

    st.subheader("Paso 2. ¿Qué tan bien funciona?")

    with st.spinner("Probando el programa..."):
        resultado = evaluar_modelo(datos, columnas)

    st.markdown(
        f"Probamos el programa con **{resultado['total']} clientes**, "
        "siempre preguntándole por clientes que no había visto mientras aprendía."
    )

    diferencia = (resultado["aciertos"] - resultado["adivinando"]) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Acierta", f"{resultado['aciertos'] * 100:.0f}%",
                  help="De cada 100 clientes, en cuántos acertó si compraría o no.")
    with col2:
        st.metric("Acertaría sin programa", f"{resultado['adivinando'] * 100:.0f}%",
                  help="Si siempre dijéramos lo más común (sin mirar al cliente).")
    with col3:
        st.metric("Ventaja del programa", f"{diferencia:+.0f} puntos")

    estado = veredicto(resultado["separacion"])
    if estado == "no_sirve":
        st.error(
            "**Conclusión: con estos datos el programa NO es confiable.** "
            "Acierta casi lo mismo que adivinar sin mirar nada. "
            "No conviene decidir a quién ofrecer la venta usando estas predicciones."
        )
    elif estado == "poco":
        st.warning(
            "**Conclusión: ayuda un poco, pero con cautela.** "
            "Sirve como una pista, no como la decisión final."
        )
    else:
        st.success(
            "**Conclusión: el programa distingue bien a los clientes.** "
            "Aun así, revise los resultados con criterio antes de actuar."
        )

    # Aciertos y errores en lenguaje de negocio
    st.markdown("**¿En qué acertó y en qué se equivocó?**")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("✅ Dijo que compraría, y sí compró", resultado["tp"])
        st.metric("✅ Dijo que no compraría, y no compró", resultado["tn"])
    with c2:
        st.metric("❌ Dijo que compraría, pero no compró", resultado["fp"],
                  help="Costo: gastamos esfuerzo de marketing en alguien que no compra.")
        st.metric("❌ Dijo que no compraría, pero sí compró", resultado["fn"],
                  help="Costo: dejamos de atender a alguien interesado.")

    # ----------------------------------
    # PASO 3. ¿QUÉ DATOS PESAN MÁS?
    # ----------------------------------

    st.subheader("Paso 3. ¿Qué datos pesan más en la decisión?")

    if estado == "no_sirve":
        st.info(
            "No mostramos esta sección porque el programa no supera al azar: "
            "los \"datos más importantes\" serían una coincidencia y podrían llevar a "
            "una mala decisión de marketing."
        )
    else:
        pesos = pd.Series({
            CARACTERISTICAS[c][0]: max(v, 0)
            for c, v in resultado["importancia"].items()
        })
        if pesos.sum() > 0:
            pesos = (pesos / pesos.sum() * 100).round(0).sort_values(ascending=False)
            st.bar_chart(pesos, horizontal=True)
            st.caption(
                "Porcentaje de influencia en la predicción. "
                "Influir no significa que sea la causa de la compra."
            )
        else:
            st.info("Ningún dato destacó claramente sobre los demás.")

    # ----------------------------------
    # PASO 4. PROBAR CON UN CLIENTE
    # ----------------------------------

    st.subheader("Paso 4. Pruebe con un cliente")

    st.markdown("Describa a un cliente y vea qué estima el programa.")

    modelo_final = entrenar_modelo_final(datos, columnas)
    perfil = {}

    columnas_pantalla = st.columns(3)
    for i, columna in enumerate(columnas):
        nombre, tipo = CARACTERISTICAS[columna]
        with columnas_pantalla[i % 3]:
            if tipo == "texto":
                opciones = sorted(datos[columna].dropna().unique())
                perfil[columna] = st.selectbox(nombre, opciones, key=f"perfil_{columna}")
            else:
                minimo, maximo = int(datos[columna].min()), int(datos[columna].max())
                perfil[columna] = st.slider(
                    nombre, minimo, maximo,
                    int(datos[columna].median()),
                    key=f"perfil_{columna}"
                )

    probabilidad = modelo_final.predict_proba(pd.DataFrame([perfil]))[0][1]

    st.metric(
        "Probabilidad estimada de que compre",
        f"{probabilidad * 100:.0f}%"
    )
    st.progress(float(probabilidad))

    if estado == "no_sirve":
        st.caption(
            "⚠️ Tome este número como ilustrativo: el programa no es confiable con estos datos."
        )

    # ----------------------------------
    # LIMITACIONES
    # ----------------------------------

    with st.expander("Antes de usar estos resultados, tenga en cuenta"):
        st.markdown("""
        - **Pocos clientes:** con 250 registros los resultados pueden cambiar mucho con otros datos.
        - **Datos ficticios:** este ejercicio es para aprender; no es una predicción real.
        - **Falta información clave:** no sabemos cuántas veces visitó la tienda, si abrió correos
          o qué le interesa. Esos datos probablemente ayudarían más que el color o la talla.
        - **Predecir no es explicar:** que un dato pese no quiere decir que cause la compra.
        - **Siempre con criterio humano:** una persona debe revisar antes de actuar.
        """)
