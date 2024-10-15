import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Fungsi untuk memuat dataset
@st.cache_data
def muat_dataset():
# Memuat dataset Kualitas Anggur Merah
    data_anggur = pd.read_csv('winequality-red.csv')
    
# Memuat dataset Wawasan Nilai Rumah
    data_rumah = pd.read_csv('homevalue.csv')
    
    return data_anggur, data_rumah

# Fungsi untuk memproses data
def proses_data(df, kolom_target):
    X = df.drop(kolom_target, axis=1)
    y = df[kolom_target]
    
    X_latih, X_uji, y_latih, y_uji = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_latih, X_uji, y_latih, y_uji

# Fungsi untuk membuat dan melatih model Regresi Linear
def latih_regresi_linear(X_latih, y_latih):
    model = LinearRegression()
    model.fit(X_latih, y_latih)
    return model

# Fungsi untuk mengevaluasi model
def evaluasi_model(model, X_uji, y_uji):
    prediksi = model.predict(X_uji)
    mse = mean_squared_error(y_uji, prediksi)
    return mse

# Fungsi untuk memprediksi data baru
def prediksi_data_baru(model, data_baru):
    prediksi = model.predict(data_baru)
    return prediksi

# Fungsi untuk menampilkan formulir input manual
def formulir_input_manual(kolom):
    data_input = {}
    st.write("### Masukkan Data Secara Manual:")
    for kol in kolom:
        data_input[kol] = st.number_input(f"Masukkan nilai untuk {kol}", step=0.01)
    return pd.DataFrame([data_input])

# Aplikasi Streamlit Utama
def main():
    st.title("Aplikasi Model Regresi Linear dengan Input Manual dan Otomatis")

# Memuat dataset
    st.write("Memuat dataset...")
    data_anggur, data_rumah = muat_dataset()
    
# Sidebar untuk pemilihan dataset
    pilihan_dataset = st.sidebar.selectbox("Pilih Dataset", ("Kualitas Anggur Merah", "Wawasan Nilai Rumah"))

    if pilihan_dataset == "Kualitas Anggur Merah":
        st.header("Dataset Kualitas Anggur Merah")
        st.write(data_anggur.head())
        
        X_latih_anggur, X_uji_anggur, y_latih_anggur, y_uji_anggur = proses_data(data_anggur, 'quality')
        model_anggur = latih_regresi_linear(X_latih_anggur, y_latih_anggur)
        mse_anggur = evaluasi_model(model_anggur, X_uji_anggur, y_uji_anggur)
        st.write(f"Kinerja Model (MSE): {mse_anggur}")

# Opsi untuk memilih input manual atau data acak
        opsi_input = st.radio("Pilih metode input:", ("Hasilkan Data Acak", "Masukkan Data Secara Manual"))

        if opsi_input == "Hasilkan Data Acak":
            st.subheader("Hasilkan Data Baru & Prediksi")
            jumlah_sampel_anggur = st.slider("Jumlah data baru yang dibutuhkan :", 1, 20, 10)
            data_anggur_baru = pd.DataFrame(
                np.random.uniform(X_latih_anggur.min(), X_latih_anggur.max(), size=(jumlah_sampel_anggur, X_latih_anggur.shape[1])),
                columns=X_latih_anggur.columns
            )
            st.write("Data Baru yang Dihasilkan:")
            st.write(data_anggur_baru)
            prediksi_anggur = prediksi_data_baru(model_anggur, data_anggur_baru)
            st.write("Prediksi untuk Data Baru:")
            st.write(prediksi_anggur)

        elif opsi_input == "Masukkan Data Secara Manual":
            data_manual_anggur = formulir_input_manual(X_latih_anggur.columns)
            st.write("Input Data Manual:")
            st.write(data_manual_anggur)
            prediksi_anggur_manual = prediksi_data_baru(model_anggur, data_manual_anggur)
            st.write("Prediksi untuk Data yang Dimasukkan:")
            st.write(prediksi_anggur_manual)

    elif pilihan_dataset == "Wawasan Nilai Rumah":
        st.header("Dataset Wawasan Nilai Rumah")
        st.write(data_rumah.head())
        
        X_latih_rumah, X_uji_rumah, y_latih_rumah, y_uji_rumah = proses_data(data_rumah, 'House_Price')  # Mengasumsikan 'House_Price' sebagai kolom target
        model_rumah = latih_regresi_linear(X_latih_rumah, y_latih_rumah)
        mse_rumah = evaluasi_model(model_rumah, X_uji_rumah, y_uji_rumah)
        st.write(f"Kinerja Model (MSE): {mse_rumah}")

# Opsi untuk memilih input manual atau data acak
        opsi_input = st.radio("Pilih metode input:", ("Hasilkan Data Acak", "Masukkan Data Secara Manual"))

        if opsi_input == "Hasilkan Data Acak":
            st.subheader("Hasilkan Data Baru & Prediksi")
            jumlah_sampel_rumah = st.slider("Jumlah data baru yang dibutuhkan", 1, 20, 10)
            data_rumah_baru = pd.DataFrame(
                np.random.uniform(X_latih_rumah.min(), X_latih_rumah.max(), size=(jumlah_sampel_rumah, X_latih_rumah.shape[1])),
                columns=X_latih_rumah.columns
            )
            st.write("Data Baru yang Dihasilkan:")
            st.write(data_rumah_baru)
            prediksi_rumah = prediksi_data_baru(model_rumah, data_rumah_baru)
            st.write("Prediksi untuk Data Baru:")
            st.write(prediksi_rumah)

        elif opsi_input == "Masukkan Data Secara Manual":
            data_manual_rumah = formulir_input_manual(X_latih_rumah.columns)
            st.write("Input Data Manual:")
            st.write(data_manual_rumah)
            prediksi_rumah_manual = prediksi_data_baru(model_rumah, data_manual_rumah)
            st.write("Prediksi untuk Data yang Dimasukkan:")
            st.write(prediksi_rumah_manual)

if __name__ == "__main__":
    main()
