from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
with open('modelo_entrenado.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del formulario como JSON
    data = request.get_json()

    # Extraer los valores del JSON
    edad = float(data['age'])
    heart_rate = float(data['heartRate'])
    systolic_bp = float(data['systolicBP'])
    diastolic_bp = float(data['diastolicBP'])
    blood_sugar = float(data['bloodSugar'])
    ckmb = float(data['ckmb'])
    troponin = float(data['troponin'])

    # Crear un arreglo con los datos del usuario para el modelo
    input_data = np.array([[edad, heart_rate, systolic_bp, diastolic_bp, blood_sugar, ckmb, troponin]])

    # Hacer la predicción
    prediccion = model.predict(input_data)

    # Preparar la respuesta
    resultado = 'Riesgo de infarto' if prediccion[0] == 1 else 'Sin riesgo de infarto'

    # Devolver el resultado como JSON
    return jsonify(prediccion=resultado)

if __name__ == "__main__":
    app.run(debug=True)