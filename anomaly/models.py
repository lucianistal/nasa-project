"""Implementaciones de métodos de detección solicitados.

Contiene:
- z_score_scores
- pca_reconstruction_scores
- isolation_forest_score
- oneclass_svm_score
- dense autoencoder (Keras)
- lstm_autoencoder (Keras)
- tcn_vae (simplified conv1d VAE)
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Import TensorFlow lazily and handle missing / broken installs gracefully
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, losses
    TF_AVAILABLE = True
except Exception:
    tf = None
    layers = None
    models = None
    losses = None
    TF_AVAILABLE = False

def z_score_scores(df, feature_cols):
    # per-row max absolute z across features
    z = np.abs((df[feature_cols] - df[feature_cols].mean())/df[feature_cols].std(ddof=0))
    return z.max(axis=1).values, z.values

def pca_reconstruction_scores(train_df, df, feature_cols, n_components=0.95):
    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit(train_df[feature_cols])
    recon = pca.inverse_transform(pca.transform(df[feature_cols]))
    rec_err = np.mean((df[feature_cols].values - recon)**2, axis=1)
    return rec_err, (df[feature_cols].values - recon)

def isolation_forest_score(train_df, df, feature_cols):
    iso = IsolationForest(contamination='auto', random_state=42)
    iso.fit(train_df[feature_cols])
    # anomaly score: negative of score_samples (so higher = more anomalous)
    raw = -iso.score_samples(df[feature_cols])
    return raw, None

def oneclass_svm_score(train_df, df, feature_cols):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_df[feature_cols])
    X = scaler.transform(df[feature_cols])
    oc = OneClassSVM(nu=0.01, kernel='rbf', gamma='scale')
    oc.fit(Xtr)
    raw = -oc.decision_function(X)  # larger = more anomalous
    return raw, None

def build_dense_autoencoder(input_dim, latent_dim=8):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available. Install a compatible TensorFlow build to use deep models.")
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(max(32, input_dim*2), activation='relu')(inp)
    x = layers.Dense(64, activation='relu')(x)
    z = layers.Dense(latent_dim, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(z)
    x = layers.Dense(max(32, input_dim*2), activation='relu')(x)
    out = layers.Dense(input_dim, activation='linear')(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

def dense_autoencoder_train_predict(train_df, df, feature_cols, latent_dim=8, epochs=30, batch_size=256):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available. Install a compatible TensorFlow build to use deep models.")
    Xtr = train_df[feature_cols].values
    X = df[feature_cols].values
    model = build_dense_autoencoder(Xtr.shape[1], latent_dim)
    model.fit(Xtr, Xtr, epochs=epochs, batch_size=batch_size, verbose=0)
    recon_tr = model.predict(Xtr)
    recon = model.predict(X)
    rec_err = np.mean((X - recon)**2, axis=1)
    feature_err = (X - recon)
    return rec_err, feature_err, model

def build_lstm_autoencoder(window_size, n_features, latent_dim=32):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available. Install a compatible TensorFlow build to use deep models.")
    inp = layers.Input(shape=(window_size, n_features))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.LSTM(latent_dim, return_sequences=False)(x)
    x = layers.RepeatVector(window_size)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(n_features))(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

def lstm_autoencoder_train_predict(train_windows, windows, epochs=20, batch_size=64):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available. Install a compatible TensorFlow build to use deep models.")
    # train_windows: array (n_samples, w, f)
    model = build_lstm_autoencoder(train_windows.shape[1], train_windows.shape[2])
    model.fit(train_windows, train_windows, epochs=epochs, batch_size=batch_size, verbose=0)
    recon = model.predict(windows)
    rec_err = np.mean((windows - recon)**2, axis=(1,2))
    return rec_err, (windows - recon), model

def build_tcn_vae(window_size, n_features, latent_dim=16):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available. Install a compatible TensorFlow build to use deep models.")
    # Simplified Conv1D VAE (TCN-like dilation can be added later)
    inp = layers.Input(shape=(window_size, n_features))
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inp)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    def sampling(args):
        z_mean, z_log_var = args
        eps = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
        return z_mean + tf.exp(0.5*z_log_var) * eps

    z = layers.Lambda(sampling)([z_mean, z_log_var])
    x = layers.Dense(window_size * 32, activation='relu')(z)
    x = layers.Reshape((window_size, 32))(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    out = layers.Conv1D(n_features, 1, padding='same', activation='linear')(x)
    model = models.Model(inp, out)

    # VAE loss
    recon_loss = tf.reduce_mean(tf.math.squared_difference(inp, out))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    model.add_loss(recon_loss + kl_loss)
    model.compile(optimizer='adam')
    return model

def tcn_vae_train_predict(train_windows, windows, epochs=20, batch_size=64):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available. Install a compatible TensorFlow build to use deep models.")
    model = build_tcn_vae(train_windows.shape[1], train_windows.shape[2])
    model.fit(train_windows, epochs=epochs, batch_size=batch_size, verbose=0)
    recon = model.predict(windows)
    rec_err = np.mean((windows - recon)**2, axis=(1,2))
    return rec_err, (windows - recon), model
