
import streamlit as st
import pandas as pd
import joblib
import os

# Set page configuration for Streamlit
st.set_page_config(page_title="Riego Inteligente", page_icon="💧")

# --- Funciones de Carga de Modelos y Preprocesadores ---
# Define the folder path where models are saved
# IMPORTANT: Adjust this path if your models are in a different location in your GitHub repo!
folder_path = '.' # Assuming 'Despliegue' folder is in the root of your repo

@st.cache_resource
def load_resources():
    try:
        # Cargar el modelo
        bagging_classifier_model = joblib.load('bagging_classifier_model.joblib')

        # Cargar el OneHotEncoder
        onehot_encoder = joblib.load('onehot_encoder.joblib')

        # Cargar el MinMaxScaler
        minmax_scaler = joblib.load('minmax_scaler.joblib')

        # Cargar el LabelEncoder (para la variable objetivo)

        label_encoder = joblib.load('label_encoder.joblib')

        return bagging_classifier_model, onehot_encoder, minmax_scaler, label_encoder
    except Exception as e:
        st.error(f"Error al cargar los recursos del modelo: {e}. Asegúrese de que los archivos .joblib están en la carpeta './Despliegue' o ajuste la variable `folder_path`.")
        return None, None, None, None

bagging_classifier_model, onehot_encoder, minmax_scaler, label_encoder = load_resources()

if bagging_classifier_model is None:
    st.stop() # Stop if resources couldn't be loaded

# --- Título y Descripción de la Aplicación ---
st.title("💧 Predicción de Riego Necesario")
st.markdown("Esta aplicación predice el nivel de riego necesario (Bajo, Medio, Alto) basándose en las condiciones ambientales y del cultivo.")
st.write("---")

# --- Sidebar para Entradas del Usuario ---
st.sidebar.header("Parámetros de Entrada")

def user_input_features():
    # Adjusted slider ranges based on common sense and potential data ranges
    soil_moisture = st.sidebar.slider("Humedad del Suelo (0.0 - 1.0)", 0.0, 1.0, 0.45, step=0.01)
    temperature_c = st.sidebar.slider("Temperatura (°C)", -10.0, 45.0, 25.0, step=0.1)
    wind_speed_kmh = st.sidebar.slider("Velocidad del Viento (km/h)", 0.0, 100.0, 10.5, step=0.1)
    crop_growth_stage = st.sidebar.selectbox("Etapa de Crecimiento del Cultivo", ('Sowing', 'Vegetative', 'Flowering', 'Harvest'))
    mulching_used = st.sidebar.selectbox("Uso de Acolchado", ('Yes', 'No'))

    data = {
        'Humedad del Suelo': soil_moisture,
        'Temperatura (°C)': temperature_c,
        'Velocidad del Viento (km/h)': wind_speed_kmh,
        'Etapa de Crecimiento del Cultivo': crop_growth_stage,
        'Uso de Acolchado': mulching_used
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.sidebar.write("---")

# --- Mostrar Entradas del Usuario (formato lista) ---
st.subheader("Parámetros de Entrada Seleccionados")

# Display input features as a list
for feature, value in df_input.iloc[0].items():
    st.markdown(f"- **{feature}**: {value}")
st.write("---")

# --- Función de Preprocesamiento ---
@st.cache_data
def preprocess_input(input_df, _onehot_enc, _minmax_sc):
    # Revertir los nombres de las columnas para el preprocesamiento si se cambiaron para la visualización
    # Crea un DataFrame temporal con los nombres de columnas originales esperados por el modelo
    temp_df = pd.DataFrame({
        'Soil_Moisture': input_df['Humedad del Suelo'],
        'Temperature_C': input_df['Temperatura (°C)'],
        'Wind_Speed_kmh': input_df['Velocidad del Viento (km/h)'],
        'Crop_Growth_Stage': input_df['Etapa de Crecimiento del Cultivo'],
        'Mulching_Used': input_df['Uso de Acolchado']
    })

    numerical_features = ['Soil_Moisture', 'Temperature_C', 'Wind_Speed_kmh']
    categorical_features_for_ohe = ['Crop_Growth_Stage', 'Mulching_Used']

    processed_df = temp_df.copy()

    try:
        # One-Hot Encoding
        onehot_encoded_features = _onehot_enc.transform(processed_df[categorical_features_for_ohe])
        onehot_feature_names = _onehot_enc.get_feature_names_out(categorical_features_for_ohe)
        onehot_df = pd.DataFrame(onehot_encoded_features, columns=onehot_feature_names, index=processed_df.index)
    except ValueError as e:
        st.error(f"Error en One-Hot Encoding: {e}. Asegúrese de que las categorías de entrada coincidan con las del entrenamiento.")
        return None

    try:
        # MinMaxScaler
        scaled_numerical_features = _minmax_sc.transform(processed_df[numerical_features])
        scaled_numerical_df = pd.DataFrame(scaled_numerical_features, columns=numerical_features, index=processed_df.index)
    except ValueError as e:
        st.error(f"Error en MinMaxScaler: {e}. Asegúrese de que los rangos de entrada sean válidos.")
        return None

    # Combine preprocessed features
    processed_data_dict = {
        'Soil_Moisture': scaled_numerical_df['Soil_Moisture'].iloc[0],
        'Temperature_C': scaled_numerical_df['Temperature_C'].iloc[0],
        'Wind_Speed_kmh': scaled_numerical_df['Wind_Speed_kmh'].iloc[0],
    }

    # Handle 'Mulching_Used_Encoded'
    if 'Mulching_Used_Yes' in onehot_df.columns:
        processed_data_dict['Mulching_Used_Encoded'] = onehot_df['Mulching_Used_Yes'].iloc[0]
    elif 'Mulching_Used_No' in onehot_df.columns:
        processed_data_dict['Mulching_Used_Encoded'] = 1 - onehot_df['Mulching_Used_No'].iloc[0]
    else:
        processed_data_dict['Mulching_Used_Encoded'] = 0
        st.warning("Advertencia: No se encontró 'Mulching_Used_Yes' ni 'Mulching_Used_No' en la salida del OneHotEncoder. Estableciendo a 0.")

    # Add one-hot encoded features for Crop_Growth_Stage
    crop_growth_stage_ohe_cols = [col for col in onehot_feature_names if col.startswith('Crop_Growth_Stage_')]
    for col in crop_growth_stage_ohe_cols:
        if col in onehot_df.columns:
            processed_data_dict[col] = onehot_df[col].iloc[0]
        else:
            processed_data_dict[col] = 0 # Fill with 0 if a category was not seen in training OHE

    # Define the final order of features expected by the model
    final_model_features = [
        'Soil_Moisture', 'Temperature_C', 'Wind_Speed_kmh',
        'Mulching_Used_Encoded',
        'Crop_Growth_Stage_Flowering', 'Crop_Growth_Stage_Harvest',
        'Crop_Growth_Stage_Sowing', 'Crop_Growth_Stage_Vegetative'
    ]

    # Ensure all final_model_features are present in processed_data_dict
    # and create the DataFrame in the correct order.
    X_processed = pd.DataFrame([processed_data_dict], columns=final_model_features)

    return X_processed

# --- Prediction Logic ---
if st.button("Predecir Nivel de Riego"):
    X_processed = preprocess_input(df_input, onehot_encoder, minmax_scaler)

    if X_processed is not None:
        # st.subheader("Datos Preprocesados (para el modelo)")
        # st.write(X_processed)
        # st.write("---")
        try:
            prediction_numeric = bagging_classifier_model.predict(X_processed)
            prediction_label = label_encoder.inverse_transform(prediction_numeric)

            st.subheader("Resultado de la Predicción")
            st.success(f"El nivel de riego necesario es: **{prediction_label[0]}**")
        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}")
