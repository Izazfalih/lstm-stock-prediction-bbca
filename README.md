# Prediksi Harga Close Saham BBCA - Stacked LSTM

## Deskripsi

Project ini menggunakan algoritma Deep Learning, yaitu **Stacked Long Short-Term Memory (LSTM)**, untuk memprediksi harga penutupan (Close Price) saham BBCA berdasarkan data historis.

## Persyaratan Sistem (Library)

Pastikan library berikut sudah terinstall:

- numpy  
- pandas  
- matplotlib  
- scikit-learn  
- tensorflow  
- yfinance  

## Cara Install Library

Jalankan perintah berikut di terminal:

```
pip install numpy pandas matplotlib scikit-learn tensorflow yfinance
```


## Alur Sistem

1. Mengambil data saham dari Yahoo Finance  
2. Preprocessing dan normalisasi data  
3. Membuat sequence data (sliding window)  
4. Training model LSTM  
5. Evaluasi model (RMSE dan MAPE)  
6. Prediksi harga saham berikutnya  

## Cara Menjalankan Program

1. Pastikan koneksi internet aktif  
2. Buka terminal di folder project  
3. Jalankan perintah berikut:

```
python bbca_lstm_project.py
```

## Output yang Dihasilkan

- Grafik Loss (Training vs Validation) dalam bentuk popup  
- Visualisasi perbandingan data aktual dan prediksi  
- Nilai evaluasi model (RMSE dan MAPE) pada terminal  
- Prediksi harga saham untuk hari berikutnya  
- Dataset saham dalam bentuk file CSV  
- Model tersimpan dengan nama `lstm_bbca_fullhistory.h5`  