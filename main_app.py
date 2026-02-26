"""
🌸 Clasificador IRIS - App Pedagógica con Streamlit
Autor: Generado con Claude
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, ConfusionMatrixDisplay
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ─────────────────────────────────────────────
# Config página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🌸 Clasificador IRIS",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS personalizado
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-bottom: 8px;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.05rem;
        margin-bottom: 24px;
    }
    .metric-card {
        background: #f8f9ff;
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #6a11cb;
        margin-bottom: 8px;
    }
    .info-box {
        background: #eef2ff;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.95rem;
    }
    .prediction-result {
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Carga de datos
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    df['target'] = iris.target
    return df, iris

df, iris = load_data()
SPECIES_COLORS = {"setosa": "#FF6B6B", "versicolor": "#4ECDC4", "virginica": "#45B7D1"}
SPECIES_EMOJI = {"setosa": "🌺", "versicolor": "🌼", "virginica": "🌸"}

# ─────────────────────────────────────────────
# Sidebar – Configuración
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/320px-Iris_versicolor_3.jpg",
             use_container_width=True, caption="Iris versicolor")
    st.markdown("## ⚙️ Configuración")

    st.markdown("### 🤖 Algoritmo")
    algorithm = st.selectbox(
        "Selecciona el clasificador",
        ["K-Nearest Neighbors (KNN)", "Decision Tree", "SVM", "Random Forest", "Logistic Regression"],
        index=0
    )

    # Hiperparámetros dinámicos
    st.markdown("### 🎛️ Hiperparámetros")
    params = {}

    if algorithm == "K-Nearest Neighbors (KNN)":
        params["n_neighbors"] = st.slider("Número de vecinos (k)", 1, 20, 5)
        params["weights"] = st.selectbox("Pesos", ["uniform", "distance"])
        params["metric"] = st.selectbox("Métrica de distancia", ["euclidean", "manhattan", "minkowski"])

    elif algorithm == "Decision Tree":
        params["max_depth"] = st.slider("Profundidad máxima", 1, 15, 3)
        params["criterion"] = st.selectbox("Criterio", ["gini", "entropy"])
        params["min_samples_split"] = st.slider("Min samples split", 2, 20, 2)

    elif algorithm == "SVM":
        params["C"] = st.slider("Parámetro C (regularización)", 0.01, 10.0, 1.0, 0.01)
        params["kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
        params["gamma"] = st.selectbox("Gamma", ["scale", "auto"])

    elif algorithm == "Random Forest":
        params["n_estimators"] = st.slider("Número de árboles", 10, 200, 100, 10)
        params["max_depth"] = st.slider("Profundidad máxima", 1, 15, 5)
        params["max_features"] = st.selectbox("Max features", ["sqrt", "log2", None])

    elif algorithm == "Logistic Regression":
        params["C"] = st.slider("Inverso de regularización (C)", 0.01, 10.0, 1.0, 0.01)
        params["max_iter"] = st.slider("Max iteraciones", 100, 1000, 200, 50)
        params["solver"] = st.selectbox("Solver", ["lbfgs", "newton-cg", "saga"])

    st.markdown("### 🧪 División de datos")
    test_size = st.slider("Tamaño del conjunto de prueba (%)", 10, 40, 20) / 100
    random_state = st.number_input("Semilla aleatoria", 0, 100, 42)
    normalize = st.checkbox("Normalizar datos (StandardScaler)", value=True)

    st.markdown("---")
    train_btn = st.button("🚀 Entrenar modelo", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# Título
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🌸 Clasificador IRIS Interactivo</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Aprende Machine Learning explorando y entrenando clasificadores en tiempo real</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Tabs principales
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Exploración de Datos",
    "🤖 Entrenamiento & Métricas",
    "📈 Visualización de Desempeño",
    "🔮 Predicción en Vivo",
    "📚 Guía Pedagógica"
])

# ═══════════════════════════════════════════════════════
# TAB 1 – EXPLORACIÓN DE DATOS
# ═══════════════════════════════════════════════════════
with tab1:
    st.header("📊 Exploración del Dataset IRIS")
    st.markdown('<div class="info-box">El dataset IRIS contiene 150 muestras de 3 especies de flores con 4 características cada una. Es el "Hello World" del Machine Learning.</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Total muestras", 150)
    col2.metric("🔢 Características", 4)
    col3.metric("🌸 Clases", 3)
    col4.metric("⚖️ Balance", "50/50/50")

    st.subheader("Vista del Dataset")
    st.dataframe(df.style.background_gradient(subset=iris.feature_names, cmap="YlOrRd"), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución por especie")
        fig_pie = px.pie(df, names='species', color='species',
                         color_discrete_map=SPECIES_COLORS,
                         hole=0.4, title="Proporción de clases")
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Estadísticas descriptivas")
        st.dataframe(df.groupby("species")[iris.feature_names].mean().round(2).style.background_gradient(cmap="Blues"), use_container_width=True)

    st.subheader("Distribución de características (Violin Plot)")
    feature_sel = st.selectbox("Característica", iris.feature_names, key="violin_feat")
    fig_violin = px.violin(df, y=feature_sel, x='species', color='species',
                           color_discrete_map=SPECIES_COLORS, box=True, points="all",
                           title=f"Distribución de {feature_sel} por especie")
    st.plotly_chart(fig_violin, use_container_width=True)

    st.subheader("Scatter Matrix (Pairplot)")
    fig_scatter = px.scatter_matrix(df, dimensions=iris.feature_names, color='species',
                                     color_discrete_map=SPECIES_COLORS, opacity=0.7,
                                     title="Relaciones entre características")
    fig_scatter.update_traces(diagonal_visible=False, showupperhalf=False)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Mapa de correlaciones")
    corr = df[iris.feature_names].corr()
    fig_corr, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlación entre características")
    st.pyplot(fig_corr)

# ═══════════════════════════════════════════════════════
# Función para construir el modelo
# ═══════════════════════════════════════════════════════
@st.cache_data
def train_model(algorithm, params, test_size, random_state, normalize):
    X = df[iris.feature_names].values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Selección del modelo
    model_map = {
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(**params),
        "Decision Tree": DecisionTreeClassifier(**params, random_state=random_state),
        "SVM": SVC(**params, probability=True, random_state=random_state),
        "Random Forest": RandomForestClassifier(**params, random_state=random_state),
        "Logistic Regression": LogisticRegression(**params, random_state=random_state),
    }
    model = model_map[algorithm]
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    cv_scores = cross_val_score(model_map[algorithm].__class__(**params if algorithm not in ["Decision Tree","SVM","Random Forest","Logistic Regression"] else {**params, **({"random_state": random_state} if algorithm != "K-Nearest Neighbors (KNN)" else {})}),
                                X, y, cv=5, scoring='accuracy')

    return model, scaler, X_train, X_test, y_train, y_test, y_pred, acc, report, cm, cv_scores

# ─── Entrenar siempre o cuando se presiona botón ───
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

if train_btn or not st.session_state.model_trained:
    with st.spinner("Entrenando modelo..."):
        result = train_model(algorithm, params, test_size, random_state, normalize)
        (model, scaler, X_train, X_test, y_train, y_test,
         y_pred, acc, report, cm, cv_scores) = result
        st.session_state.update({
            "model": model, "scaler": scaler,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "y_pred": y_pred, "acc": acc, "report": report,
            "cm": cm, "cv_scores": cv_scores,
            "algorithm": algorithm, "model_trained": True
        })

# Recuperar del estado
model = st.session_state.get("model")
scaler = st.session_state.get("scaler")
acc = st.session_state.get("acc", 0)
report = st.session_state.get("report", {})
cm = st.session_state.get("cm")
cv_scores = st.session_state.get("cv_scores", np.array([]))
y_test = st.session_state.get("y_test", np.array([]))
y_pred = st.session_state.get("y_pred", np.array([]))
X_test = st.session_state.get("X_test", np.array([]))

# ═══════════════════════════════════════════════════════
# TAB 2 – ENTRENAMIENTO & MÉTRICAS
# ═══════════════════════════════════════════════════════
with tab2:
    st.header("🤖 Entrenamiento y Métricas")

    if model is None:
        st.info("Presiona **🚀 Entrenar modelo** en el panel lateral.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 Accuracy", f"{acc:.2%}")
        col2.metric("📊 CV Mean", f"{cv_scores.mean():.2%}")
        col3.metric("📉 CV Std", f"±{cv_scores.std():.3f}")
        col4.metric("🏋️ Train size", f"{len(st.session_state['X_train'])} muestras")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Reporte de Clasificación")
            report_df = pd.DataFrame(report).transpose()
            report_df = report_df.drop("support", axis=1, errors="ignore")
            st.dataframe(
                report_df.style
                    .format("{:.3f}")
                    .background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
                use_container_width=True
            )

        with col2:
            st.subheader("Matriz de Confusión")
            fig_cm = px.imshow(
                cm,
                x=iris.target_names.tolist(),
                y=iris.target_names.tolist(),
                color_continuous_scale="Blues",
                text_auto=True,
                title="Matriz de Confusión",
                labels=dict(x="Predicción", y="Real", color="Cantidad")
            )
            fig_cm.update_layout(height=350)
            st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("Cross-Validation por Fold")
        cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(len(cv_scores))], "Accuracy": cv_scores})
        fig_cv = px.bar(cv_df, x="Fold", y="Accuracy", color="Accuracy",
                        color_continuous_scale="Viridis", range_y=[0.8, 1.0],
                        title="Accuracy por fold (5-fold CV)")
        fig_cv.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="red",
                         annotation_text=f"Media: {cv_scores.mean():.3f}")
        st.plotly_chart(fig_cv, use_container_width=True)

        # Feature importance (si aplica)
        if hasattr(model, "feature_importances_"):
            st.subheader("Importancia de Características")
            fi = pd.DataFrame({
                "Feature": iris.feature_names,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=True)
            fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                            color="Importance", color_continuous_scale="Oranges",
                            title="Importancia de cada característica")
            st.plotly_chart(fig_fi, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 3 – VISUALIZACIÓN DE DESEMPEÑO
# ═══════════════════════════════════════════════════════
with tab3:
    st.header("📈 Visualización de Desempeño")

    if model is None:
        st.info("Presiona **🚀 Entrenar modelo** en el panel lateral.")
    else:
        st.subheader("Comparación Real vs. Predicho (Test Set)")
        comp_df = pd.DataFrame({
            "Index": range(len(y_test)),
            "Real": [iris.target_names[i] for i in y_test],
            "Predicho": [iris.target_names[i] for i in y_pred],
            "Correcto": y_test == y_pred
        })
        fig_comp = px.scatter(
            comp_df, x="Index", y="Real", color="Correcto",
            symbol="Predicho",
            color_discrete_map={True: "#2ecc71", False: "#e74c3c"},
            title="Predicciones en el conjunto de prueba (✅ correcto / ❌ error)"
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Curva de aprendizaje
        st.subheader("Curva de Aprendizaje")
        from sklearn.model_selection import learning_curve as lc_fn

        X_all = df[iris.feature_names].values
        y_all = df['target'].values
        if normalize and scaler is not None:
            X_all_scaled = scaler.transform(X_all)
        else:
            X_all_scaled = X_all

        try:
            train_sizes, train_scores, val_scores = lc_fn(
                model, X_all_scaled, y_all,
                cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring="accuracy"
            )
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(
                x=train_sizes, y=train_scores.mean(axis=1),
                mode="lines+markers", name="Train Accuracy",
                line=dict(color="#6a11cb", width=2),
                error_y=dict(type='data', array=train_scores.std(axis=1), visible=True)
            ))
            fig_lc.add_trace(go.Scatter(
                x=train_sizes, y=val_scores.mean(axis=1),
                mode="lines+markers", name="Validation Accuracy",
                line=dict(color="#2575fc", width=2),
                error_y=dict(type='data', array=val_scores.std(axis=1), visible=True)
            ))
            fig_lc.update_layout(
                title="Curva de Aprendizaje",
                xaxis_title="Tamaño del conjunto de entrenamiento",
                yaxis_title="Accuracy", yaxis_range=[0.5, 1.05],
                legend=dict(orientation="h")
            )
            st.plotly_chart(fig_lc, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo calcular la curva de aprendizaje: {e}")

        # Scatter 2D de características vs predicciones
        st.subheader("Visualización 2D: Predicciones vs. Realidad")
        col1, col2 = st.columns(2)
        feat_x = col1.selectbox("Eje X", iris.feature_names, index=0, key="scatter2d_x")
        feat_y = col2.selectbox("Eje Y", iris.feature_names, index=1, key="scatter2d_y")

        test_df = pd.DataFrame(
            X_test if not normalize else X_test,
            columns=iris.feature_names if not normalize else [f"{f} (norm)" for f in iris.feature_names]
        )
        # Usar índices originales
        X_test_orig = df[iris.feature_names].values[
            np.array_split(np.arange(len(df)), 1)[0][:len(X_test)]
        ]
        test_df_orig = df.iloc[-len(X_test):][iris.feature_names].copy()
        test_df_orig["Real"] = [iris.target_names[i] for i in y_test]
        test_df_orig["Predicho"] = [iris.target_names[i] for i in y_pred]
        test_df_orig["Correcto"] = y_test == y_pred

        fig_2d = px.scatter(
            test_df_orig, x=feat_x, y=feat_y,
            color="Real", symbol="Correcto",
            color_discrete_map=SPECIES_COLORS,
            symbol_map={True: "circle", False: "x"},
            title=f"Distribución: {feat_x} vs {feat_y}",
            hover_data=["Real", "Predicho"]
        )
        st.plotly_chart(fig_2d, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 4 – PREDICCIÓN EN VIVO
# ═══════════════════════════════════════════════════════
with tab4:
    st.header("🔮 Predicción en Vivo")
    st.markdown('<div class="info-box">Ajusta los valores de las 4 características y el modelo predecirá la especie de flor en tiempo real.</div>', unsafe_allow_html=True)

    if model is None:
        st.info("Primero entrena el modelo en el panel lateral.")
    else:
        col_slider, col_result = st.columns([1, 1])

        with col_slider:
            st.subheader("🎚️ Ingresa los valores")
            sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
            sepal_width  = st.slider("Sepal width (cm)",  1.5, 5.0, 3.0, 0.1)
            petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.3, 0.1)
            petal_width  = st.slider("Petal width (cm)",  0.1, 2.6, 1.3, 0.1)

            st.markdown("---")
            st.markdown("**O carga un ejemplo del dataset:**")
            sample_idx = st.number_input("Índice de muestra (0-149)", 0, 149, 0)
            if st.button("📥 Cargar muestra"):
                sepal_length = df.iloc[sample_idx]["sepal length (cm)"]
                sepal_width  = df.iloc[sample_idx]["sepal width (cm)"]
                petal_length = df.iloc[sample_idx]["petal length (cm)"]
                petal_width  = df.iloc[sample_idx]["petal width (cm)"]
                st.rerun()

        with col_result:
            st.subheader("🌸 Resultado de la predicción")
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            if scaler:
                input_scaled = scaler.transform(input_data)
            else:
                input_scaled = input_data

            prediction = model.predict(input_scaled)[0]
            species_name = iris.target_names[prediction]
            emoji = SPECIES_EMOJI[species_name]
            color = SPECIES_COLORS[species_name]

            st.markdown(
                f'<div class="prediction-result" style="background:{color}22;border:3px solid {color}">'
                f'{emoji} <span style="color:{color}">{species_name.upper()}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Probabilidades
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_scaled)[0]
                prob_df = pd.DataFrame({
                    "Especie": iris.target_names,
                    "Probabilidad": probs
                })
                fig_prob = px.bar(
                    prob_df, x="Especie", y="Probabilidad",
                    color="Especie", color_discrete_map=SPECIES_COLORS,
                    title="Probabilidad por clase", range_y=[0, 1],
                    text_auto=".2%"
                )
                fig_prob.update_layout(showlegend=False)
                st.plotly_chart(fig_prob, use_container_width=True)
            else:
                decision = model.decision_function(input_scaled)[0]
                dec_df = pd.DataFrame({"Clase": iris.target_names, "Score": decision})
                fig_dec = px.bar(dec_df, x="Clase", y="Score", color="Clase",
                                 color_discrete_map=SPECIES_COLORS,
                                 title="Decision Function Score")
                st.plotly_chart(fig_dec, use_container_width=True)

            # Comparación con muestra real
            real_species = df.iloc[sample_idx]["species"]
            st.markdown(f"**Especie real del índice {sample_idx}:** `{real_species}`")
            match = real_species == species_name
            st.success("✅ ¡Predicción correcta!" if match else "❌ Predicción incorrecta.")

        # Tabla de entrada
        st.subheader("📋 Resumen de entrada")
        input_table = pd.DataFrame({
            "Característica": iris.feature_names,
            "Valor ingresado": [sepal_length, sepal_width, petal_length, petal_width],
            "Media global": df[iris.feature_names].mean().values.round(2),
            "Diferencia": (np.array([sepal_length, sepal_width, petal_length, petal_width]) - df[iris.feature_names].mean().values).round(2)
        })
        st.dataframe(input_table.style.background_gradient(subset=["Diferencia"], cmap="RdBu_r"), use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 5 – GUÍA PEDAGÓGICA
# ═══════════════════════════════════════════════════════
with tab5:
    st.header("📚 Guía Pedagógica")

    with st.expander("🌸 ¿Qué es el dataset IRIS?", expanded=True):
        st.markdown("""
        El **Dataset IRIS** fue introducido por Ronald Fisher en 1936 y es uno de los datasets más famosos en estadística y Machine Learning.
        
        - Contiene **150 muestras** de flores de iris.
        - Hay **3 clases** (especies): *Setosa*, *Versicolor*, *Virginica*.
        - Cada muestra tiene **4 características**:
          - `sepal length` – largo del sépalo (cm)
          - `sepal width` – ancho del sépalo (cm)
          - `petal length` – largo del pétalo (cm)
          - `petal width` – ancho del pétalo (cm)
        """)

    with st.expander("🤖 ¿Cómo funcionan los algoritmos?"):
        algo_info = {
            "K-Nearest Neighbors (KNN)": """
            **KNN** clasifica un punto nuevo buscando los **k vecinos más cercanos** en el espacio de características.
            La clase más frecuente entre esos vecinos es la predicción.
            - **k pequeño** → modelo complejo, posible sobreajuste
            - **k grande** → modelo simple, posible subajuste
            """,
            "Decision Tree": """
            Un **Árbol de Decisión** divide el espacio de características haciendo preguntas binarias sucesivas.
            - Intuitivo y fácil de interpretar
            - Sensible a sobreajuste si la profundidad es muy alta
            - La pureza se mide con **Gini** o **Entropía**
            """,
            "SVM": """
            **SVM (Support Vector Machine)** encuentra el hiperplano que maximiza el margen entre clases.
            - Muy efectivo en espacios de alta dimensión
            - El parámetro **C** controla el trade-off entre margen y errores
            - Los **kernels** (rbf, poly) permiten clasificación no lineal
            """,
            "Random Forest": """
            **Random Forest** es un **ensemble** de múltiples Árboles de Decisión entrenados en subconjuntos aleatorios.
            - Reduce el sobreajuste respecto a un solo árbol
            - Proporciona **importancia de características**
            - Más robusto y preciso que un árbol individual
            """,
            "Logistic Regression": """
            La **Regresión Logística** modela la probabilidad de pertenecer a cada clase usando una función sigmoide.
            - Modelo lineal, rápido e interpretable
            - El parámetro **C** controla la regularización
            - Funciona bien cuando las clases son linealmente separables
            """
        }
        for name, desc in algo_info.items():
            st.markdown(f"### {name}")
            st.markdown(desc)
            st.markdown("---")

    with st.expander("📊 ¿Cómo interpretar las métricas?"):
        st.markdown("""
        | Métrica | Fórmula | ¿Qué mide? |
        |---------|---------|------------|
        | **Accuracy** | TP+TN / Total | % de predicciones correctas |
        | **Precision** | TP / (TP+FP) | De los que dije clase X, ¿cuántos son realmente X? |
        | **Recall** | TP / (TP+FN) | De los que son clase X, ¿cuántos detecté? |
        | **F1-Score** | 2×P×R/(P+R) | Balance entre Precision y Recall |
        | **CV Score** | Promedio de k-folds | Robustez del modelo ante diferentes particiones |
        
        **Matriz de Confusión:** Muestra cuántas muestras de cada clase real fueron clasificadas en cada clase predicha.
        """)

    with st.expander("🧪 Experimentos sugeridos"):
        st.markdown("""
        1. **Efecto de k en KNN**: Prueba valores de k desde 1 hasta 20. ¿Cuál da mejor CV score?
        2. **Overfitting en Decision Tree**: Aumenta la profundidad máxima. ¿Qué pasa con train vs. validation?
        3. **Kernels en SVM**: Compara `linear`, `rbf` y `poly`. ¿Cuál funciona mejor?
        4. **Normalización**: Activa/desactiva el escalado. ¿Afecta a todos los algoritmos igual?
        5. **Tamaño del test set**: Cambia de 10% a 40%. ¿Cómo cambia la estabilidad del accuracy?
        """)

st.markdown("---")
st.caption("🌸 App creada con Streamlit + Scikit-learn | Dataset IRIS - Fisher (1936) | Generada con Claude")
