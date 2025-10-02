import pandas as pd
import numpy as np
import tensorflow as tf
from model.domain.LeGrandParams import params
from model.domain.PINN import PINN_model
import json
    
def map_frontend_data(form_data):
    """
    Mapea los datos del frontend al formato requerido por el modelo.
    
    Parameters:
    -----------
    form_data : FormData
        Datos del formulario frontend
    
    Returns:
    --------
    dict : Diccionario con los datos mapeados y validados
    """
    try:
        # Mapear sexo
        sexo_map = {
            'masculino': 0,
            'hombre': 0,
            'h': 0,
            'femenino': 1,
            'mujer': 1,
            'm': 1
        }
        sexo_lower = form_data.sexo.lower().strip()
        sexo_valor = sexo_map.get(sexo_lower)
        
        if sexo_valor is None:
            raise ValueError(f"Valor de sexo no válido: '{form_data.sexo}'. Use 'masculino' o 'femenino'")
        
        # Convertir strings a float
        mapped_data = {
            'CW': float(form_data.cw),
            'Edad años': float(form_data.edad),
            'Sexo': sexo_valor,
            'SA': float(form_data.sa),
            'V': float(form_data.v),
            'Dioptrias': float(form_data.dioptrias)
        }
        
        return mapped_data
        
    except ValueError as e:
        raise ValueError(f"Error al convertir los datos: {e}")

def predict_from_frontend(form_data):
    """
    Función principal para realizar predicciones desde datos del frontend.
    
    Parameters:
    -----------
    form_data : FormData
        Datos del formulario frontend
    
    Returns:
    --------
    dict : Diccionario con las predicciones y datos de entrada procesados
    """
    try:
        # Mapear datos del frontend
        mapped_data = map_frontend_data(form_data)
        print(f"Datos mapeados: {mapped_data}")
        
        # Realizar predicción
        predictions = load_model_and_predict(mapped_data)
        
        # Preparar respuesta
        response = {
            'input': {
                'CW': mapped_data['CW'],
                'Edad': mapped_data['Edad años'],
                'Sexo': 'Masculino' if mapped_data['Sexo'] == 0 else 'Femenino',
                'SA': mapped_data['SA'],
                'V': mapped_data['V'],
                'Dioptrias': mapped_data['Dioptrias']
            },
            'predictions': predictions
        }
        
        return response
        
    except Exception as e:
        raise Exception(f"Error en la predicción: {e}")
    

def load_model_and_predict(input_data, model_dir='model/weights/'):
    """
    Carga el modelo PINN entrenado y realiza predicciones.
    
    Parameters:
    -----------
    input_data : dict or pd.DataFrame
        Datos de entrada con las siguientes variables:
        - CW: Cámara de White (mm)
        - Edad años: Edad del paciente (años)
        - Sexo: Sexo del paciente (0=Hombre, 1=Mujer)
        - SA: Superficie anterior (mm²)
        - V: Volumen (mm³)
        - Dioptrias: Dioptrías (D)
    
    Returns:
    --------
    predictions : dict
        Diccionario con las predicciones:
        - AXL: Longitud axial (mm)
        - LENS: Espesor del cristalino (mm)
        - Paq: Espesor de la córnea (mm)
        - RaK: Radio anterior de la córnea (mm)
        - ACD: Profundidad de la cámara anterior (mm)
    """
    
    # Convertir input_data a DataFrame si es un diccionario
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()
    
    # Verificar que todas las columnas necesarias están presentes
    required_columns = ['CW', 'Edad años', 'Sexo', 'SA', 'V', 'Dioptrias']
    missing_columns = [col for col in required_columns if col not in input_df.columns]
    
    if missing_columns:
        raise ValueError(f"Faltan las siguientes columnas: {missing_columns}")
    
    # Preparar datos de entrada
    X_input = input_df[required_columns].values
    
    # Cargar el modelo
    # Nota: Necesitamos crear una instancia del modelo con los parámetros correctos
    # y luego cargar los pesos guardados
    
    # Crear dummy data para inicializar el modelo (no se usará para predicción)
    X_dummy = np.zeros((1, 6))
    y_dummy = np.zeros((1, 5))
    
    # Inicializar el modelo
    model = PINN_model(
        params,
        X_dummy, X_dummy,
        y_dummy, y_dummy,
        lambda_data=1,
        lambda_physics=0.7
    )
    
    # Construir el modelo ejecutando una predicción dummy
    _ = model(tf.constant(X_dummy, dtype=tf.float32))
    
    # Cargar los pesos guardados
    try:
        model = cargar_modelo_completo(model, save_dir=model_dir)
    except FileNotFoundError as e:
        raise Exception(
            f"Error: No se encontraron los archivos del modelo en '{model_dir}'.\n"
            f"Asegúrate de haber guardado el modelo con 'guardar_modelo_completo()' después del entrenamiento.\n"
            f"Detalle: {e}"
        )
    except Exception as e:
        raise Exception(f"Error al cargar el modelo: {e}")
    
    # Normalizar los datos de entrada usando la media y std del entrenamiento
    X_input_norm = (X_input - model.X_mean.numpy()) / model.X_std.numpy()
    X_input_tf = tf.constant(X_input_norm, dtype=tf.float32)
    
    # Realizar predicción
    y_pred_norm = model(X_input_tf, training=False)
    y_pred = y_pred_norm * model.y_std + model.y_mean
    y_pred_np = y_pred.numpy()
    
    # Nombres de las variables de salida
    output_names = ["AXL", "LENS", "Paq", "RaK", "ACD"]
    
    # Crear diccionario con las predicciones
    predictions = {}
    for i, nombre in enumerate(output_names):
        if len(y_pred_np) == 1:
            predictions[nombre] = float(y_pred_np[0, i])
        else:
            predictions[nombre] = y_pred_np[:, i].tolist()
    
    return predictions


def cargar_modelo_completo(model, save_dir='model/weights/'):
    """
    Carga el modelo y sus estadísticas de normalización
    """
    import tensorflow as tf
    
    # 1. Cargar pesos
    model.load_weights(f'{save_dir}best_model.weights.h5')
    
    # 2. Cargar estadísticas de normalización
    with open(f'{save_dir}normalization_stats.json', 'r') as f:
        stats = json.load(f)
    
    # Restaurar como tensores de TensorFlow
    model.X_mean = tf.constant(stats['X_mean'], dtype=tf.float32)
    model.X_std = tf.constant(stats['X_std'], dtype=tf.float32)
    model.y_mean = tf.constant(stats['y_mean'], dtype=tf.float32)
    model.y_std = tf.constant(stats['y_std'], dtype=tf.float32)
    
    print(f"Modelo completo cargado desde {save_dir}")
    return model