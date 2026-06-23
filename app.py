from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Memuat model dan scaler asli dari Colab
# Note: Jika nanti kamu punya file pkl khusus (misal: model_dt.pkl), kamu tinggal load di sini.
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Menangkap pilihan model dari dropdown HTML
    selected_model = request.form.get('model_choice')
    
    # 2. Menangkap hanya 8 input nilai medis (mengabaikan nilai dari dropdown)
    fitur_medis = [
        float(request.form.get('Pregnancies')),
        float(request.form.get('Glucose')),
        float(request.form.get('BloodPressure')),
        float(request.form.get('SkinThickness')),
        float(request.form.get('Insulin')),
        float(request.form.get('BMI')),
        float(request.form.get('DiabetesPedigreeFunction')),
        float(request.form.get('Age'))
    ]
    
    final_features = [np.array(fitur_medis)]
    
    # 3. Menyamakan skala data menggunakan scaler
    prediction_scaler = scaler.transform(final_features)
    
    # 4. Melakukan prediksi menggunakan model (0 atau 1)
    # (Jika nanti ada model_dt atau model_rf, kamu bisa pakai percabangan if selected_model == 'dt')
    prediction = model.predict(prediction_scaler)
    
    if prediction[0] == 1:
        res_val = "Diabetes"
    else:
        res_val = "No Diabetes"
        
    # Menampilkan hasil prediksi lengkap dengan nama model yang dipilih
    nama_model_tampilan = selected_model.upper() if selected_model else "MODEL"
    return render_template('index.html', prediction_text=f'Hasil Prediksi ({nama_model_tampilan}): {res_val}')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)