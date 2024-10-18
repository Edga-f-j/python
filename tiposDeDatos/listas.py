# lista = [ 11,22,33,44,55,66,77,88,99,22,33,22]
# lista2 = ["alejandro", "alberto", "cristiano", "artemisa", "poseidon"]
# print(lista)
# # elemento = input("ingrese el valor a agregar en la lista: ")
# # lista.remove(float(elemento))
# # print(lista)

# lista.sort(reverse=False)
# print(lista)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Crear dataset simulado
data = {
    'tamaño': [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
    'habitaciones': [3, 3, 4, 4, 5, 5, 4, 3, 4, 5],
    'ubicacion': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    'precio': [300000, 320000, 340000, 360000, 380000, 400000, 420000, 440000, 460000, 480000]
}
df = pd.DataFrame(data)

# Preparar datos
X = df[['tamaño', 'habitaciones', 'ubicacion']]
y = df['precio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#####  ESCALAR DATOS #####
scaler = StandardScaler()

# Escalar las características
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




# Entrenar modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones y evaluar
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Error Absoluto Medio (MAE):", mae)
print("Error Cuadrático Medio (MSE):", mse)
print("Coeficiente de Determinación (R²):", r2)

# Visualización
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label="Precio Real", marker='o')
plt.plot(range(len(y_pred)), y_pred, label="Precio Predicho", marker='x')
plt.xlabel("Ejemplos")
plt.ylabel("Precio")
plt.title("Comparación de Precios Reales vs Predichos")
plt.legend()
plt.show()

