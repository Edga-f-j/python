import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# parte para guardar los datos en la variabe "df"
df = pd.read_csv("archivoss/tatakae.csv")




# Seleccionar las columnas que vamos a usar como características (X) y la columna target (y)
X = df[['Age', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']]
y = df['Result']  # Suponiendo que esta columna tiene 0 o 1 para representar infarto/no infarto

# Dividir los datos en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#####  ESCALAR DATOS #####
scaler = StandardScaler()

# Escalar las características
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




# Crear el modelo
model = LogisticRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)



# Hacer predicciones con el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')

# Mostrar la matriz de confusión
print('Matriz de confusión:')
print(confusion_matrix(y_test, y_pred))

# Mostrar un reporte de clasificación detallado
print('Reporte de clasificación:')
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label="Precio Real", marker='o')
plt.plot(range(len(y_pred)), y_pred, label="Precio Predicho", marker='x')
plt.xlabel("Ejemplos")
plt.ylabel("Precio")
plt.title("Comparación de Precios Reales vs Predichos")
plt.legend()
plt.show()


# plt.plot(Attack, yougn, label='Jovenes', color='blue', linestyle='--',linewidth=2)
# plt.plot(Attack, old, label='Adultos', color='green', linestyle='-', linewidth=2)
# plt.xlabel('Eje X')
# plt.ylabel('Eje Y')

# plt.title('TITULO')
# plt.show()








# X = df[['Age', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']]
# y = df['Result']  # Suponiendo que esta columna tiene 0 o 1 para representar infarto/no infarto

# # Dividir los datos en conjunto de entrenamiento (80%) y prueba (20%)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # Entrenar modelo
# model = LinearRegression()
# model.fit(X_train, y_train)
# # Hacer predicciones y evaluar
# y_pred = model.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)

# r2 = r2_score(y_test, y_pred)


# print("Error Absoluto Medio (MAE):", mae)

# print("Coeficiente de Determinación (R²):", r2)

# # Visualización
# plt.figure(figsize=(10, 6))
# plt.plot(range(len(y_test)), y_test, label="Precio Real", marker='o')
# plt.plot(range(len(y_pred)), y_pred, label="Precio Predicho", marker='x')
# plt.xlabel("Ejemplos")
# plt.ylabel("Precio")
# plt.title("Comparación de Precios Reales vs Predichos")
# plt.legend()
# plt.show()




# Attack = df["Troponin"]

# Adultos = df[(df['Age'] >= 41) & (df['Age'] <= 70)]
# Jovenes = df[(df['Age'] >= 26) & (df['Age'] <= 40)]



#########################################################
######################### EDAD  #########################

# Clasificación de personas jóvenes (<= 40 años) y mayores (> 40 años)
# Clasificar personas como jóvenes (<= 40) o mayores (> 40)
# df['grupo_edad'] = df['Age'].apply(lambda x: 'Joven' if x <= 40 else 'Mayor')


# # Agrupar por grupo de edad y resultado (infarto)
# df_agrupado = df.groupby(['grupo_edad', 'Result']).size().unstack(fill_value=0)

# # Crear gráfica de barras
# df_agrupado.plot(kind='barh')

# # Añadir etiquetas y título
# plt.title('Cantidad de Personas Jóvenes vs Mayores con Infartos')
# plt.xlabel('Grupo de Edad')
# plt.ylabel('Cantidad de Personas')

# # Mostrar gráfica
# plt.show()


###########################################################
######################### GENERO  #########################



# df['Grupo_genero'] = df['Gender'].apply(lambda x: "Men" if x ==1 else "Woman")

# df_agrupacion2 = df.groupby(['Grupo_genero','Result']).size().unstack(fill_value=0)

# df_agrupacion2.plot(kind='bar')

# plt.title('Cantidad de Hombres vs Mujeres con Infartos')
# plt.xlabel('Grupo de Edad')
# plt.ylabel('Cantidad de Personas')

# # Mostrar gráfica
# plt.show()


###########################################################
######################### Troponin  #########################

# df['Grupo_Troponin'] = df['Troponin'].apply(lambda x: "Alto" if x<=0.004  else "Bajo")

# df_agrupacion2 = df.groupby(['Grupo_Troponin','Result']).size().unstack(fill_value=0)

# df_agrupacion2.plot(kind='bar')

# plt.title('Personas con Troponin bajo y Troponin Alto con y  sin Infartos')
# plt.xlabel('Grupo de Edad')
# plt.ylabel('Cantidad de Personas')

# # Mostrar gráfica
# plt.show()






# Attack = df["Troponin"]
# yougn = Jovenes["Age"]
# old = Adultos["Age"]

# #convenciones
# #Muestra los datos en consola
# # print(Attack)




# plt.plot(Attack, yougn, label='Jovenes', color='blue', linestyle='--',linewidth=2)
# plt.plot(Attack, old, label='Adultos', color='green', linestyle='-', linewidth=2)
# plt.xlabel('Eje X')
# plt.ylabel('Eje Y')

# plt.title('TITULO')
# plt.show()