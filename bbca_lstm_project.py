import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from datetime import date

# Parameters
TICKER = "BBCA.JK"
START = "2000-01-01"                     # mulai dari tahun 2000
END = date.today().isoformat()           # sampai hari ini (contoh: '2025-12-16')
LOOK_BACK = 60
TEST_RATIO = 0.2
BATCH_SIZE = 32
EPOCHS = 30
RANDOM_SEED = 42

# 1) Download data (Yahoo Finance)
df = yf.download(TICKER, start=START, end=END)
df.to_csv('Data_Saham_BBCA_Mentah.csv')
print("Data berhasil disimpan ke file 'Data_Saham_BBCA_Mentah.csv'")
if df.empty:
    raise SystemExit("Data kosong. Pastikan koneksi internet dan ticker benar (BBCA.JK).")

# 2) Ensure chronological order (oldest -> newest)
df = df.sort_index(ascending=True)
data = df[['Close']].dropna()
print(f"Loaded {len(data)} rows. Date range: {data.index.min().date()} to {data.index.max().date()}")

# 3) Scale (MinMax 0-1)
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(data.values)  # shape (n,1)

# 4) Create sequences (sliding window)
def create_sequences(series, look_back):
    X, y = [], []
    for i in range(look_back, len(series)):
        X.append(series[i-look_back:i, 0])
        y.append(series[i, 0])
    X = np.array(X)
    y = np.array(y)
    return X, y

X, y = create_sequences(scaled, LOOK_BACK)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 5) Chronological train/test split
n_samples = X.shape[0]
n_test = int(n_samples * TEST_RATIO)
n_train = n_samples - n_test

X_train = X[:n_train]; y_train = y[:n_train]
X_test  = X[n_train:]; y_test  = y[n_train:]

print(f"Samples total: {n_samples}, train: {len(X_train)}, test: {len(X_test)}")

# 6) Build LSTM model (2 LSTM layers -> Dense(25) -> Dense(1))
import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(LOOK_BACK, 1)),
    LSTM(64, return_sequences=False),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# 7) Train with EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es],
    verbose=2
)

# 8) Predict & inverse transform
pred_test = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
pred_test_inv = scaler.inverse_transform(pred_test.reshape(-1,1))

# 9) Evaluate (RMSE, MAPE)
rmse = math.sqrt(mean_squared_error(y_test_inv, pred_test_inv))
mape = mean_absolute_percentage_error(y_test_inv, pred_test_inv) * 100
print(f"\nEvaluation on TEST set:")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f} %")

# 10) Save model
model.save("lstm_bbca_fullhistory.h5")
print("Model saved to lstm_bbca_fullhistory.h5")

# GRAFIK LOSS
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss Evaluation (Train vs Validation)')
plt.ylabel('Mean Squared Error (MSE)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show() 


# 11) Plot results with train/test boundary
# reconstruct train & test true series for plotting alignment
dates = data.index[LOOK_BACK:]  # because first LOOK_BACK days have no y
train_dates = dates[:n_train]
test_dates  = dates[n_train:]

# invert train predictions for plotting (optional)
pred_train = model.predict(X_train)
pred_train_inv = scaler.inverse_transform(pred_train.reshape(-1,1))
y_train_inv = scaler.inverse_transform(y_train.reshape(-1,1))

plt.figure(figsize=(14,6))
plt.plot(train_dates, y_train_inv, label="Train Actual", linewidth=1)
plt.plot(train_dates, pred_train_inv, label="Train Pred", linewidth=1, alpha=0.8)
plt.plot(test_dates, y_test_inv, label="Test Actual", linewidth=1)
plt.plot(test_dates, pred_test_inv, label="Test Pred", linewidth=1, alpha=0.9)
# vertical line separating train/test
plt.axvline(x=test_dates[0], color='k', linestyle='--', linewidth=1, label='Train/Test split')
plt.title(f"{TICKER} Close Price - LSTM Predictions (History {START} → {END})")
plt.xlabel("Date")
plt.ylabel("Price (IDR)")
plt.legend()
plt.tight_layout()
plt.show()


# 12) Next-day prediction (1-step ahead) using last available window
last_window = scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)
next_scaled = model.predict(last_window)
next_price = scaler.inverse_transform(next_scaled.reshape(-1,1))[0,0]
last_date = data.index.max().date()
print(f"Last available date in dataset: {last_date}")
print(f"Prediksi Harga Close untuk (next trading day) adalah ≈ IDR {next_price:,.2f}")
