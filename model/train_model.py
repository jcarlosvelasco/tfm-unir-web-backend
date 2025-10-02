#Model

#Imports
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from model.domain.LeGrandParams import params
from model.domain.PINN import PINN_model
from model.helpers import potencia_lente_correctora, potencia_superficie
import json


def train_model():
    #Load dataset
    data = pd.DataFrame(pd.read_excel("model/domain/Base Datos Pacientes.xlsx"))

    def calc(param, AXL, espesorCristalino, espesorCornea, radioAnteriorCornea, acd):
        radioPosteriorCornea = radioAnteriorCornea * 0.822

        #Potencias superficiales
        Pcornea_anterior = potencia_superficie(1.000, param['n_cornea'], radioAnteriorCornea)
        Pcornea_posterior = potencia_superficie(param['n_cornea'], param['n_acuoso'], radioPosteriorCornea)
        Pcristalino_anterior = potencia_superficie(param['n_acuoso'], param['n_cristalino'], param['R3'])
        Pcristalino_posterior = potencia_superficie (param['n_cristalino'], param['n_vitreo'], param['R4'])

        #Potencia total
        Pcornea = Pcornea_anterior + Pcornea_posterior - (espesorCornea / param['n_cornea']) * Pcornea_anterior * Pcornea_posterior
        Pcristalino = Pcristalino_anterior + Pcristalino_posterior - (espesorCristalino / param['n_cristalino']) * Pcristalino_anterior * Pcristalino_posterior

        hc = espesorCornea * (param['n_cornea'] - 1) / Pcornea
        hl = espesorCristalino * (param['n_cristalino'] - param['n_acuoso']) / Pcristalino
        dist1 = acd - hc + hl

        P_total_2S = Pcornea + Pcristalino - (dist1 / param['n_acuoso']) * Pcornea * Pcristalino

        distancia_lente_correctora = 0.0012

        d1 = AXL - (1.91 / 1000)

        # Potencia deseada para enfocar en la retina
        P_deseada = param['n_vitreo'] / d1

        # Potencia de la lente correctora (d = distancia desde la lente a la retina)
        d = distancia_lente_correctora  # en metros
        P_lente = potencia_lente_correctora(P_total_2S, P_deseada, d)

        return P_lente

    data["Dioptrias"] = data.apply(
        lambda row: calc(params, (row['AXL'] / 1000), (row['LENS'] / 1000), (row['Paq'] / 1000), (row['RaK'] / 1000), (row['ACD'] / 1000)),
        axis=1
    )

    #Prepare data


    #Convertir la columna Eje K2 a tipo numérico
    data['Eje K2'] = pd.to_numeric(data['Eje K2'], errors='coerce')

    #Eliminar filas con valores NaN
    data.dropna(inplace=True)

    #Eliminar fecha de nacimiento y días ya que representa lo mismo que 'Edad años'
    data.drop(columns=["F. Nac"], inplace=True)
    data.drop(columns=["Días"], inplace=True)

    #Eliminar la variable "F. Ex."
    data.drop(columns=["F. Ex."], inplace=True)

    #Codificar la columna sexo a valor numérico
    data['Sexo'] = data['Sexo'].replace({'H': 0, 'H\'': 0, 'M': 1, 'M\'': 1})

    #Codificar la columna OD/OI a valor numérico
    data['OD/OI'] = data['OD/OI'].replace({'OD': 0, 'OI': 1})


    def clean_dataset(data):
        #No tiene mucho sentido buscar outliers en la edad
        not_available_columns = ['Edad años']

        cleaned_data = data.copy()
        for col in data.columns:
            if col in cleaned_data.columns:
                if cleaned_data[col].dtype in ['float64', 'int64'] and col not in not_available_columns:
                    # Calcular cuartiles
                    q1 = cleaned_data[col].quantile(0.25)
                    q3 = cleaned_data[col].quantile(0.75)
                    iqr = q3 - q1

                    # Calcular límites
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    # Contar outliers antes de eliminar
                    outliers_count = len(cleaned_data[(cleaned_data[col] < lower_bound) |
                                                    (cleaned_data[col] > upper_bound)])

                    # Filtrar outliers
                    cleaned_data = cleaned_data[(cleaned_data[col] >= lower_bound) &
                                            (cleaned_data[col] <= upper_bound)]

                    print(f"Columna '{col}': Eliminados {outliers_count} outliers "
                        f"(límites: {lower_bound:.2f} - {upper_bound:.2f})")

        print(f"\nDataset original: {len(data)} filas")
        print(f"Dataset limpio: {len(cleaned_data)} filas")
        print(f"Filas eliminadas: {len(data) - len(cleaned_data)} ({(len(data) - len(cleaned_data))/len(data)*100:.1f}%)")
        return cleaned_data

    data = clean_dataset(data)

    #Eliminar K1 y K2 -> correlación de casi 1 con Km
    data.drop(columns=["K1", "K2"], inplace=True)

    #Eliminar Eje ó Eje K1 -> correlación de 1
    data.drop(columns=["Eje"], inplace=True)

    #Eliminar Km -> correlación de 1 con RaK
    data.drop(columns=["Km"], inplace=True)

    parametros_entrada = ['CW', 'Edad años', 'Sexo', 'SA', 'V', 'Dioptrias']
    parametros_target = ["AXL", "LENS", "Paq", "RaK", "ACD"]

    X = data[parametros_entrada].values
    y = data[parametros_target].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    model = PINN_model(
        params,
        X_train, X_val,
        y_train, y_val,
        lambda_data=1,
        lambda_physics=0.7
    )

    model.fit_PINN(
        num_epochs=10000,
        print_every=100,
        save_path='model/weights/best_model.weights.h5',
        patience = 350
    )

    guardar_modelo_completo(model)
    
def guardar_modelo_completo(model, save_dir='model/weights/'):
    """
    Guarda el modelo y todas las estadísticas necesarias para predicción reproducible
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Guardar pesos del modelo
    model.save_weights(f'{save_dir}best_model.weights.h5')
    
    # 2. Guardar estadísticas de normalización
    stats = {
        'X_mean': model.X_mean.numpy().tolist(),
        'X_std': model.X_std.numpy().tolist(),
        'y_mean': model.y_mean.numpy().tolist(),
        'y_std': model.y_std.numpy().tolist()
    }
    
    with open(f'{save_dir}normalization_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Modelo completo guardado en {save_dir}")
    print("  - best_model.weights.h5 (pesos)")
    print("  - normalization_stats.json (estadísticas)")
    
if __name__ == "__main__":
    train_model()


