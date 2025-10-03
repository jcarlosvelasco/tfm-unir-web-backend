import tensorflow as tf
from keras import layers

from model.helpers import potencia_lente_correctora, potencia_superficie


class PINN_model(tf.keras.Model):
    def __init__(self, parametros, X_train, X_val, y_train, y_val, lambda_data=1.0, lambda_physics=0.5, **kwargs):
        super(PINN_model, self).__init__(**kwargs)
        # Valores de ponderación del error PINNs
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.parametros = parametros

        # Datos con los que se va a realizar el entrenamiento
        self.X_train = tf.constant(X_train, dtype=tf.float32)
        self.X_val = tf.constant(X_val, dtype=tf.float32)
        self.y_train = tf.constant(y_train, dtype=tf.float32)
        self.y_val = tf.constant(y_val, dtype=tf.float32)

        # Normalización de los datos
        self.X_mean = tf.reduce_mean(self.X_train, axis=0)
        self.X_std = tf.math.reduce_std(self.X_train, axis=0)
        self.y_mean = tf.reduce_mean(self.y_train, axis=0)
        self.y_std = tf.math.reduce_std(self.y_train, axis=0)

        self.X_train_norm = (self.X_train - self.X_mean) / self.X_std
        self.X_val_norm = (self.X_val - self.X_mean) / self.X_std
        self.y_train_norm = (self.y_train - self.y_mean) / self.y_std
        self.y_val_norm = (self.y_val - self.y_mean) / self.y_std

        # Hiperparametros del entrenamiento
        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate)
        self.activation = 'tanh'

        # Arquitectura de la red neuronal
        self.input_dim = self.X_train.shape[1]
        self.output_dim = self.y_train.shape[1]
        
        self.dense0 = layers.Dense(self.input_dim, activation=self.activation, name='dense0')
        self.dense1 = layers.Dense(128, activation=self.activation, name='dense1')
        self.dropout1 = layers.Dropout(0.3, name='dropout1')
        self.dense2 = layers.Dense(128, activation=self.activation, name='dense2')
        self.dropout2 = layers.Dropout(0.3, name='dropout2')
        self.dense3 = layers.Dense(64, activation=self.activation, name='dense3')
        self.dense4 = layers.Dense(32, activation=self.activation, name='dense4')
        self.output_layer = layers.Dense(self.output_dim, name='output_layer')

        # Construir el modelo con la forma de entrada correcta
        self.build((None, self.input_dim))

        # Datos último entrenamiento
        self.history = {
            'train_total': [],
            'val_total': [],
            'train_data': [],
            'val_data': [],
            'train_physics': [],
            'val_physics': []
        }

    def build(self, input_shape):
        """Implementa el método build correctamente"""
        if not self.built:
            super(PINN_model, self).build(input_shape)
            # Las capas se construirán automáticamente en la primera llamada

    def call(self, features, training=False):
        """
        Forward pass del modelo con control explícito del modo de entrenamiento
        """
        x = self.dense0(features)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.output_layer(x)

    def calculo_error_refractivo_tensores(self, axl_tensor, lens_tensor, paq_tensor, rak_tensor, acd_tensor):
        """Calculo del error refractivo para modelo tensorflow"""
        radioPosteriorCornea = rak_tensor * 0.822

        #Potencias superficiales
        Pcornea_anterior = potencia_superficie(1.000, self.parametros['n_cornea'], rak_tensor)
        Pcornea_posterior = potencia_superficie(self.parametros['n_cornea'], self.parametros['n_acuoso'], radioPosteriorCornea)
        Pcristalino_anterior = potencia_superficie(self.parametros['n_acuoso'], self.parametros['n_cristalino'], self.parametros['R3'])
        Pcristalino_posterior = potencia_superficie (self.parametros['n_cristalino'], self.parametros['n_vitreo'], self.parametros['R4'])

        #Potencia total
        Pcornea = Pcornea_anterior + Pcornea_posterior - (paq_tensor / self.parametros['n_cornea']) * Pcornea_anterior * Pcornea_posterior
        Pcristalino = Pcristalino_anterior + Pcristalino_posterior - (lens_tensor / self.parametros['n_cristalino']) * Pcristalino_anterior * Pcristalino_posterior

        hc = paq_tensor * (self.parametros['n_cornea'] - 1) / Pcornea
        hl = lens_tensor * (self.parametros['n_cristalino'] - self.parametros['n_acuoso']) / Pcristalino
        dist1 = acd_tensor - hc + hl

        P_total_2S = Pcornea + Pcristalino - (dist1 / self.parametros['n_acuoso']) * Pcornea * Pcristalino

        distancia_lente_correctora = 0.0012
        d2 = axl_tensor - (1.91 / 1000)

        # Potencia deseada para enfocar en la retina
        P_deseada = self.parametros['n_vitreo'] / d2

        # Potencia de la lente correctora (d = distancia desde la lente a la retina)
        d = distancia_lente_correctora  # en metros
        P_lente = potencia_lente_correctora(P_total_2S, P_deseada, d)
        return P_lente

    def total_loss(self, l_data, l_physics):
        """Función de pérdidad total PINNs"""
        return self.lambda_data * l_data + self.lambda_physics * l_physics

    @tf.function
    def train_step(self, x_scaled, y_scaled, x_orig):
        """Función step del entrenamiento"""
        with tf.GradientTape() as tape:
            y_pred_scaled = self(x_scaled, training=True)
            l_data = tf.reduce_mean(tf.square(y_pred_scaled - y_scaled))

            y_pred_orig = y_pred_scaled * self.y_std + self.y_mean
            axl, lens, paq, rak, acd = [tf.squeeze(t) for t in tf.split(y_pred_orig, 5, axis=1)]
            dioptrias_real = tf.squeeze(x_orig[:, -1])

            error_refractivo = self.calculo_error_refractivo_tensores(axl / 1000, lens / 1000, paq / 1000, rak / 1000, acd / 1000)
            l_physics = tf.reduce_mean(tf.square(error_refractivo - dioptrias_real))

            total_loss = self.total_loss(l_data, l_physics)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return total_loss, l_data, l_physics

    @tf.function
    def val_step(self, x_scaled, y_scaled, x_orig):
        """Paso de validación del entrenamiento"""
        y_pred_scaled = self(x_scaled, training=False)
        l_data_val = tf.reduce_mean(tf.square(y_pred_scaled - y_scaled))

        y_pred_orig = y_pred_scaled * self.y_std + self.y_mean
        axl, lens, paq, rak, acd = [tf.squeeze(t) for t in tf.split(y_pred_orig, 5, axis=1)]
        dioptrias_real = tf.squeeze(x_orig[:, -1])
        error_refractivo = self.calculo_error_refractivo_tensores(axl / 1000, lens / 1000, paq / 1000, rak / 1000, acd / 1000)
        l_physics_val = tf.reduce_mean(tf.square(error_refractivo - dioptrias_real))
        loss_val = self.total_loss(l_data_val, l_physics_val)
        return loss_val, l_data_val, l_physics_val

    def fit_PINN(self, num_epochs, print_every=100, patience=100):
        """Entrenamiento del modelo con guardado automático de parámetros de normalización"""
        best_val_loss = float('inf')
        best_epoch = 0
        epochs_without_improvement = 0
        best_weights = None
        
        history = {
            'train_total': [],
            'val_total': [],
            'train_data': [],
            'val_data': [],
            'train_physics': [],
            'val_physics': []
        }

        for epoch in range(num_epochs):
            loss_train, l_data_train, l_physics_train = self.train_step(
                self.X_train_norm, self.y_train_norm, self.X_train
            )
            loss_val, l_data_val, l_physics_val = self.val_step(
                self.X_val_norm, self.y_val_norm, self.X_val
            )

            # Guardar historial
            history['train_total'].append(loss_train.numpy())
            history['val_total'].append(loss_val.numpy())
            history['train_data'].append(l_data_train.numpy())
            history['val_data'].append(l_data_val.numpy())
            history['train_physics'].append(l_physics_train.numpy())
            history['val_physics'].append(l_physics_val.numpy())

            if loss_val.numpy() < best_val_loss:
                best_val_loss = loss_val.numpy()
                best_epoch = epoch + 1
                best_weights = self.get_weights()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Train: {loss_train:.4f} (Data: {l_data_train:.4f}, Physics: {l_physics_train:.4f}) | Val: {loss_val:.4f} (Data: {l_data_val:.4f}, Physics: {l_physics_val:.4f})")

            if epochs_without_improvement >= patience:
                print(f"Early stopping en la época {epoch+1} (no mejora en {patience} épocas).")
                break

        if best_weights is not None:
            print(f"Cargando los mejores pesos de la época {best_epoch} con pérdida de validación: {best_val_loss:.4f}")
            self.set_weights(best_weights)

        print(f"Entrenamiento finalizado. Mejor modelo en época {best_epoch} con pérdida de validación: {best_val_loss:.4f}")
        self.history = history

        return history