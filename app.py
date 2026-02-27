import streamlit as st
import joblib

# Load model dan vectorizer
model = joblib.load('sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Tampilan UI Streamlit
st.set_page_config(page_title="Analisis Sentimen Rakha", page_icon="📊")
st.title("Aplikasi Analisis Sentimen Sederhana")
st.markdown("---")
st.write("Aplikasi ini memprediksi apakah ulasan bermakna **Positif** atau **Negatif**.")

# Input user
user_input = st.text_area("Masukkan teks ulasan di sini:", placeholder="Contoh: Barangnya bagus dan sampai cepat")

if st.button("Analisis Sekarang"):
    if user_input.strip() != "":
        # Transformasi input
        text_vector = tfidf.transform([user_input])
        # Prediksi
        prediction = model.predict(text_vector)[0]
        
        # Tampilkan hasil dengan warna
        if prediction == "Positif":
            st.success(f"Hasil Analisis: **{prediction}** 😊")
        else:
            st.error(f"Hasil Analisis: **{prediction}** 😡")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu!")

st.markdown("---")
st.caption("Dibuat oleh Muhamad Rakha Hadyan Pangestu - Informatika Unila")