import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import os
from InputGen import load_clones
from tensorflow.keras.utils import register_keras_serializable
import joblib
from tensorflow.keras.callbacks import CSVLogger


@register_keras_serializable()
# Custom activation function (unchanged)
def custom_activation(x):
    return 1 / (1 + tf.exp(-tf.reduce_sum(x, axis=-1, keepdims=True)))

@register_keras_serializable()
class SelfAttentionLayer(layers.Layer):
    """
    Self-attention mechanism - computes attention weights for each feature
    based on all other features in the same input.
    """

    def __init__(self, units, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W_q = layers.Dense(units, use_bias=False)
        self.W_k = layers.Dense(units, use_bias=False)
        self.W_v = layers.Dense(units, use_bias=False)

    def call(self, inputs):
        # Use a Lambda to safely expand dims inside model graph
        x = inputs if len(inputs.shape) == 3 else tf.expand_dims(inputs, axis=1)

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        attention_scores = tf.matmul(Q, K, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.units, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attended = tf.matmul(attention_weights, V)

        return tf.squeeze(attended, axis=1) if attended.shape[1] == 1 else attended

# -------------------------------
# 2. Feature-wise Attention Layer
# -------------------------------
@register_keras_serializable()
class FeatureAttentionLayer(layers.Layer):
    """
    Computes attention weights for each feature dimension.
    Perfect for your 9-dimensional input where you want to weight features.
    """

    def __init__(self, **kwargs):
        super(FeatureAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention_weights = self.add_weight(
            name='attention_weights',
            shape=(input_shape[-1],),
            initializer='glorot_uniform',
            trainable=True
        )
        super(FeatureAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        norm_weights = tf.nn.softmax(self.attention_weights)
        return inputs * norm_weights  # Element-wise feature scaling


# -------------------------------
# 3. Additive Attention Layer (Bahdanau-style)
# -------------------------------
@register_keras_serializable()
class AdditiveAttentionLayer(layers.Layer):
    """
    Additive attention mechanism (also called Bahdanau attention).
    Good for sequence-to-sequence tasks.
    """

    def __init__(self, units, **kwargs):
        super(AdditiveAttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W1 = layers.Dense(units, activation='tanh')
        self.W2 = layers.Dense(1)

    def call(self, inputs):
        x = inputs if len(inputs.shape) == 3 else tf.expand_dims(inputs, axis=1)
        score = self.W2(self.W1(x))  # (batch_size, seq_len, 1)
        weights = tf.nn.softmax(score, axis=1)
        weighted = x * weights
        return tf.reduce_sum(weighted, axis=1)  # Collapse sequence


# -------------------------------
# Attention Layer: Multi-Head
# -------------------------------
@register_keras_serializable()
class MultiHeadAttentionLayer(layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=1)

        batch_size = tf.shape(inputs)[0]
        q = self.split_heads(self.wq(inputs), batch_size)
        k = self.split_heads(self.wk(inputs), batch_size)
        v = self.split_heads(self.wv(inputs), batch_size)

        scaled_attention_logits = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))

        out = self.dense(concat_attention)
        return tf.squeeze(out, axis=1) if out.shape[1] == 1 else out

# -------------------------------
# Sampling Layer
# -------------------------------
@register_keras_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# -------------------------------
# VAE Loss Wrapper
# -------------------------------
@register_keras_serializable()
class VAELossLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        x, z_mean, z_log_var, reconstructed = inputs
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - reconstructed), axis=1))
        kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        self.add_loss(reconstruction_loss + kl_loss)
        return reconstructed

@register_keras_serializable()
class VariationalAutoencoder(Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_layer = VAELossLayer()

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return self.loss_layer([inputs, z_mean, z_log_var, reconstructed])

# -------------------------------
# ModelBuilder Class
# -------------------------------
class ModelBuilder:
    def __init__(self, input_dim=9, expanded_dim=32, latent_dim=32):
        self.input_dim = input_dim
        self.expanded_dim = expanded_dim
        self.latent_dim = latent_dim
        self.attention_type = "feature"

    def build_encoder(self):
        encoder_input = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(16, activation='relu')(encoder_input)
        x = layers.Dense(self.expanded_dim, activation='relu')(x)

        # Expand dims for attention
        if self.attention_type == "multihead":
            x_expanded = layers.Reshape((1, self.expanded_dim))(x)
            x_attn = MultiHeadAttentionLayer(self.expanded_dim, num_heads=4)(x_expanded)
            x = layers.Lambda(lambda x: tf.squeeze(x, axis=1))(x_attn)
        elif self.attention_type == "self":
            x = SelfAttentionLayer(self.expanded_dim)(x)
        elif self.attention_type == "feature":
            x = FeatureAttentionLayer()(x)
        elif self.attention_type == "additive":
            x = AdditiveAttentionLayer(self.expanded_dim)(x)

        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        return Model(encoder_input, [z_mean, z_log_var, z], name="Encoder")

    def build_decoder(self):
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(self.expanded_dim, activation='relu')(latent_inputs)
        x = layers.Dense(16, activation='relu')(x)
        out = layers.Dense(self.input_dim, activation='sigmoid')(x)
        return Model(latent_inputs, out, name="Decoder")

    def build_vae(self):
        encoder = self.build_encoder()
        decoder = self.build_decoder()
        vae = VariationalAutoencoder(encoder, decoder)
        vae.compile(optimizer='adam')
        return vae, encoder, decoder

    def build_siamese_classifier(self, encoder):
        encoder.trainable = True
        input_a = layers.Input(shape=(self.input_dim,), name="input_a")
        input_b = layers.Input(shape=(self.input_dim,), name="input_b")

        z_mean_a = encoder(input_a)
        z_mean_b = encoder(input_b)

        #merged = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([z_mean_a, z_mean_b])
        merged = tf.keras.layers.Subtract()([z_mean_a, z_mean_b])
        merged = tf.keras.layers.Activation("relu")(merged)

        x = layers.Dense(32, activation='relu')(merged)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        output = layers.Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[input_a, input_b], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

def load_csv_input(trueClone, falseClone):
    print('Loading True Clones')
    trueClone = load_clones(trueClone, True)
    print('Loading False Clones')
    falseClone = load_clones(falseClone, False)
    df = pd.concat([trueClone, falseClone], ignore_index=True)
    input_a = df.iloc[:, 0:9].values.astype(np.float32)
    input_b = df.iloc[:, 9:18].values.astype(np.float32)
    labels = df.iloc[:, 18].values.astype(np.float32)

    # Normalize
    scaler = StandardScaler()
    input_a = scaler.fit_transform(input_a)
    input_b = scaler.transform(input_b)
    return input_a, input_b, labels, scaler

def build_zmean_encoder(encoder):
    inp = encoder.input
    z_mean = encoder.get_layer("z_mean").output
    return tf.keras.Model(inp, z_mean, name="ZMeanEncoder")

def main():
    builder = ModelBuilder(input_dim=9)
    csv_path = './../Data/CSVFiles'
    csv_path2 = './../Data/CSVFiles_BackUp'
    trueCloneFileName = "CSharpPythonFeatures.csv"
    falseCloneFileName = "CSharpPythonNonCloneFeatures.csv"

    a_data, b_data, labels, scalar = load_csv_input(os.path.join(csv_path, trueCloneFileName),
                                                    os.path.join(csv_path, falseCloneFileName))

    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    split = int(0.8 * len(labels))
    a_train, a_test = a_data[indices[:split]], a_data[indices[split:]]
    b_train, b_test = b_data[indices[:split]], b_data[indices[split:]]
    y_train, y_test = labels[indices[:split]], labels[indices[split:]]

    vae, encoder, _ = builder.build_vae()

    # Callbacks for VAE
    os.makedirs("Logs", exist_ok=True)
    vae_logger = CSVLogger("Logs/CSharp_Python_vae_training_log.csv")
    print("Training VAE...")
    vae.fit(a_train, a_train,
            validation_data=(a_test, a_test),
            epochs=30, batch_size=32,
            verbose=2, callbacks=[vae_logger])

    # Save VAE summary
    with open("SaveModels/CSharpPythonVAEModelLarge_summary.txt", "w") as f:
        vae.summary(print_fn=lambda x: f.write(x + '\n'))

    z_encoder = build_zmean_encoder(encoder)
    siamese = builder.build_siamese_classifier(z_encoder)

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    siamese_logger = CSVLogger("Logs/CSharp_Python_siamese_training_log.csv")

    print("Training Siamese classifier...")
    siamese.fit([a_train, b_train], y_train,
                validation_data=([a_test, b_test], y_test),
                epochs=30, batch_size=32,
                callbacks=[early_stop, lr_scheduler, siamese_logger],
                verbose=2)

    os.makedirs("SaveModels", exist_ok=True)
    siamese.save("SaveModels/CSharpPythonVAEModelLarge.keras")
    siamese.save("SaveModels/CSharpPythonVAEModelLarge.h5")
    joblib.dump(scalar, "SaveModels/CSharpPythonScalarLarge.pkl")

    with open("SaveModels/CSharpPythonSiameseModel_summary.txt", "w") as f:
        siamese.summary(print_fn=lambda x: f.write(x + '\n'))


if __name__ == "__main__":
    main()
