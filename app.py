import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuración de la página
st.set_page_config(
    page_title="Visualizador de Encuestas Barranquilla",
    page_icon="📊",
    layout="wide"
)

# Título de la aplicación
st.title("📊 Visualizador de Encuestas y Predicciones - Barranquilla")
st.markdown("---")

# Función para cargar datos
@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv("./out/checkpoint_barranquilla.csv")
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

# Función para cargar el modelo
@st.cache_resource
def cargar_modelo():
    try:
        with open("exp6_encuestas.pkl", "rb") as f:
            modelo = pickle.load(f)
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Cargar datos y modelo
df = cargar_datos()
modelo = cargar_modelo()

# Sidebar
st.sidebar.title("Opciones de Visualización")
st.sidebar.markdown("---")

# Verificar si los datos se cargaron correctamente
if df is not None:
    # Mostrar información básica
    st.sidebar.subheader("Información del Dataset")
    st.sidebar.info(f"Número de filas: {df.shape[0]}")
    st.sidebar.info(f"Número de columnas: {df.shape[1]}")
    
    # Opciones de visualización
    opcion = st.sidebar.selectbox(
        "Seleccione una visualización",
        ["Datos", "Distribución por Ideología", "Correlación Encuesta vs Resultado", "Predicción con Modelo"]
    )
    
    # Configuración de escalado
    st.sidebar.subheader("Configuración de Escalado")
    max_valor_escalado = st.sidebar.number_input(
        "Valor máximo para escalado (0-3000 → 0-100)", 
        min_value=100, 
        max_value=5000, 
        value=3000,
        step=100,
        help="El valor máximo del modelo que se escala a 100%. Ajusta este valor si las predicciones parecen muy altas o muy bajas."
    )
    
    # Mostrar datos
    if opcion == "Datos":
        st.subheader("📋 Datos de Encuestas")
        
        # Opciones para filtrar
        años_disponibles = sorted(df["año"].unique())
        año_seleccionado = st.multiselect("Filtrar por año", años_disponibles, default=años_disponibles)
        
        ideologias_disponibles = sorted(df["IDEOLOGIA"].unique())
        ideologia_seleccionada = st.multiselect("Filtrar por ideología", ideologias_disponibles, default=ideologias_disponibles)
        
        # Aplicar filtros
        df_filtrado = df[
            (df["año"].isin(año_seleccionado)) & 
            (df["IDEOLOGIA"].isin(ideologia_seleccionada))
        ]
        
        # Mostrar datos filtrados
        st.dataframe(df_filtrado)
        
        # Opción para descargar los datos filtrados
        csv = df_filtrado.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar datos filtrados como CSV",
            data=csv,
            file_name="datos_filtrados.csv",
            mime="text/csv",
        )
    
    # Distribución por ideología
    elif opcion == "Distribución por Ideología":
        st.subheader("📊 Distribución por Ideología")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfica de barras de resultados promedio por ideología
            st.subheader("Resultado promedio por ideología")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=df, x="IDEOLOGIA", y="resultado", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_xlabel("Ideología")
            ax.set_ylabel("Resultado promedio")
            st.pyplot(fig)
        
        with col2:
            # Gráfica de barras de encuestas promedio por ideología
            st.subheader("Encuesta promedio por ideología")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=df, x="IDEOLOGIA", y="encuesta", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_xlabel("Ideología")
            ax.set_ylabel("Encuesta promedio")
            st.pyplot(fig)
            
        # Distribución de resultados por ideología (boxplot)
        st.subheader("Distribución de resultados por ideología")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x="IDEOLOGIA", y="resultado", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("Ideología")
        ax.set_ylabel("Resultado")
        st.pyplot(fig)
    
    # Correlación Encuesta vs Resultado
    elif opcion == "Correlación Encuesta vs Resultado":
        st.subheader("📈 Correlación entre Encuesta y Resultado")
        
        # Selector de ideología
        ideologia_seleccionada = st.selectbox(
            "Seleccione una ideología para la correlación",
            ["Todas"] + sorted(df["IDEOLOGIA"].unique().tolist())
        )
        
        # Filtrar por ideología seleccionada
        if ideologia_seleccionada == "Todas":
            df_plot = df
            titulo = "Correlación para todas las ideologías"
        else:
            df_plot = df[df["IDEOLOGIA"] == ideologia_seleccionada]
            titulo = f"Correlación para ideología: {ideologia_seleccionada}"
        
        # Gráfico de dispersión
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_plot, x="encuesta", y="resultado", hue="IDEOLOGIA", ax=ax)
        
        # Línea de regresión
        x = df_plot["encuesta"].values.reshape(-1, 1)
        y = df_plot["resultado"].values
        if len(x) > 1:  # Asegurar que hay suficientes datos para la regresión
            reg = LinearRegression().fit(x, y)
            x_range = np.linspace(df_plot["encuesta"].min(), df_plot["encuesta"].max(), 100).reshape(-1, 1)
            y_pred = reg.predict(x_range)
            ax.plot(x_range, y_pred, color="red", linestyle="--")
            
            # Mostrar coeficientes
            st.info(f"Coeficiente (pendiente): {reg.coef_[0]:.4f}")
            st.info(f"Intercepto: {reg.intercept_:.4f}")
            st.info(f"R²: {reg.score(x, y):.4f}")
        
        ax.set_title(titulo)
        ax.set_xlabel("Valor de Encuesta")
        ax.set_ylabel("Resultado Real")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Predicción con modelo
    elif opcion == "Predicción con Modelo":
        st.subheader("🔮 Predicción con Modelo Entrenado")
        
        if modelo is not None:
            # Información del modelo
            st.info("Modelo cargado: Regresión Lineal")
            st.write("Este modelo predice el resultado basado en la encuesta y la ideología.")
            st.warning(f"Nota: El modelo de regresión lineal genera predicciones en un rango amplio. La aplicación escala automáticamente estas predicciones al rango 0-100 (usando {max_valor_escalado} como valor máximo) para facilitar su interpretación. Puedes ajustar este valor máximo en el panel lateral.")
            
            # Formulario para ingresar datos para predicción
            with st.form("formulario_prediccion"):
                st.subheader("Ingrese los datos para la predicción:")
                
                # Datos de entrada
                valor_encuesta = st.number_input("Valor de la encuesta", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
                
                ideologia = st.selectbox(
                    "Seleccione la ideología",
                    sorted(df["IDEOLOGIA"].unique())
                )
                
                # Botón para predecir
                submit_button = st.form_submit_button(label="Predecir")
                
                if submit_button:
                    # Crear features para el modelo
                    # Crear un dataframe con una fila para la predicción
                    ideologias_posibles = [
                        "Centro Derecha", "Centro Izquierda", "Centro derecha", 
                        "Centro izquierda", "Derecha", "Izquierda"
                    ]
                    
                    features = {"encuesta": [valor_encuesta]}
                    
                    # Crear variables dummy para la ideología
                    for ideologia_posible in ideologias_posibles:
                        col_name = f"IDEOLOGIA_{ideologia_posible}"
                        features[col_name] = [1 if ideologia == ideologia_posible else 0]
                    
                    # Convertir a dataframe
                    features_df = pd.DataFrame(features)
                    
                    # Asegurar que todas las columnas del modelo estén presentes
                    for col in modelo.feature_names_in_:
                        if col not in features_df.columns:
                            features_df[col] = 0
                    
                    # Reordenar columnas para que coincidan con el modelo
                    features_df = features_df[modelo.feature_names_in_]
                    
                    # Realizar predicción
                    try:
                        # Predicción original del modelo
                        prediccion_original = modelo.predict(features_df)[0][0]
                        
                        # Escalar la predicción usando el valor máximo configurado
                        prediccion = (prediccion_original / max_valor_escalado) * 100
                        
                        # Limitar la predicción al rango [0-100] por seguridad
                        prediccion = max(0, min(100, prediccion))
                        
                        # Mostrar resultados
                        st.success(f"Predicción (escalada a 0-100): {prediccion:.2f}%")
                        st.info(f"Valor original de la predicción: {prediccion_original:.2f}")
                        
                        # Comparar con valores similares en el dataset
                        st.subheader("Comparación con datos similares:")
                        df_similar = df[df["IDEOLOGIA"] == ideologia].copy()
                        df_similar["diferencia_encuesta"] = abs(df_similar["encuesta"] - valor_encuesta)
                        df_similar = df_similar.sort_values("diferencia_encuesta").head(5)
                        
                        if not df_similar.empty:
                            st.dataframe(df_similar[["IDEOLOGIA", "encuesta", "resultado", "nombre_encuesta", "año"]])
                        else:
                            st.info("No hay datos similares para comparar.")
                            
                    except Exception as e:
                        st.error(f"Error al realizar la predicción: {e}")
            
            # Visualizar coeficientes del modelo
            st.subheader("Coeficientes del modelo")
            
            # Crear un dataframe con los coeficientes
            coefs = pd.DataFrame({
                'Feature': modelo.feature_names_in_,
                'Coeficiente': modelo.coef_[0]
            })
            
            # Ordenar por valor absoluto del coeficiente
            coefs['Abs_Coef'] = coefs['Coeficiente'].abs()
            coefs = coefs.sort_values('Abs_Coef', ascending=False)
            
            # Gráfico de barras para los coeficientes
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=coefs, x='Coeficiente', y='Feature', ax=ax)
            ax.set_title("Importancia de las características")
            ax.set_xlabel("Coeficiente")
            ax.set_ylabel("Característica")
            ax.grid(True, axis='x', alpha=0.3)
            st.pyplot(fig)
            
            # Intercepto
            st.info(f"Intercepto del modelo: {modelo.intercept_[0]:.4f}")
            
            # Visualización de predicciones vs valores reales
            st.subheader("Predicciones del modelo vs Valores reales")
            
            if not df.empty and modelo is not None:
                # Preparar features para todos los datos
                X_all = pd.get_dummies(df[["encuesta", "IDEOLOGIA"]], columns=["IDEOLOGIA"], prefix=["IDEOLOGIA"])
                
                # Asegurar que todas las columnas necesarias estén presentes
                for col in modelo.feature_names_in_:
                    if col not in X_all.columns:
                        X_all[col] = 0
                
                # Reordenar columnas para que coincidan con el modelo
                X_all = X_all[modelo.feature_names_in_]
                
                # Realizar predicciones
                y_pred = modelo.predict(X_all).flatten()
                
                # Escalar predicciones usando el valor máximo configurado
                y_pred = (y_pred / max_valor_escalado) * 100
                
                # Limitar predicciones al rango [0-100] por seguridad
                y_pred = np.clip(y_pred, 0, 100)
                
                # Crear DataFrame para visualización
                df_pred = pd.DataFrame({
                    "Valores reales (escalados)": (df["resultado"] / max_valor_escalado) * 100,
                    "Valores reales (originales)": df["resultado"],
                    "Predicciones": y_pred,
                    "IDEOLOGIA": df["IDEOLOGIA"]
                })
                
                # Gráfico de dispersión
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df_pred, x="Valores reales (escalados)", y="Predicciones", hue="IDEOLOGIA", ax=ax)
                
                # Línea diagonal (predicción perfecta)
                min_val = min(df_pred["Valores reales (escalados)"].min(), df_pred["Predicciones"].min())
                max_val = max(df_pred["Valores reales (escalados)"].max(), df_pred["Predicciones"].max())
                ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
                
                ax.set_title("Predicciones del modelo vs Valores reales (escalados a 0-100)")
                ax.set_xlabel("Valores reales (escalados)")
                ax.set_ylabel("Predicciones")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Calcular métricas de rendimiento
                # Primero escalar los valores reales al mismo rango que las predicciones
                y_real_scaled = (df["resultado"] / max_valor_escalado) * 100
                
                mse = mean_squared_error(y_real_scaled, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_real_scaled, y_pred)
                r2 = r2_score(y_real_scaled, y_pred)
                
                # Mostrar métricas
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MSE", f"{mse:.2f}")
                col2.metric("RMSE", f"{rmse:.2f}")
                col3.metric("MAE", f"{mae:.2f}")
                col4.metric("R²", f"{r2:.2f}")
        else:
            st.error("No se pudo cargar el modelo. Verifique que el archivo 'exp6_encuestas.pkl' existe.")
else:
    st.error("No se pudieron cargar los datos. Verifique que el archivo 'checkpoint_barranquilla.csv' existe.")

# Información adicional en el pie de página
st.markdown("---")
st.markdown("### Información")
st.info("""
Esta aplicación visualiza datos de encuestas y predicciones para Barranquilla.
- Los datos provienen del archivo 'checkpoint_barranquilla.csv'.
- El modelo utilizado es una regresión lineal entrenada previamente.
""")