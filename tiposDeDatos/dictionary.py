

clientes = { 
    "nombre": "max",
    "apellido": "Rodriguez",
    "Cel": 3124567890,
    "saldo":500000,
    "edad": 25
}

cajero = {
    "nombre": "pepito perez",
    "apellido": "perez",
    "edad": 20
}

# añade un elemento nuevo al diccionario
clientes["profesion"] = "Ingeniero"

# Para imprimir un valor del diccionario
valor = clientes.get("nombre")
print(valor)

# para eliminar un elemento del diccionario
del clientes["edad"]
print(clientes)

# ver las claves del diccionario
claves = clientes.keys()
print(claves)

# ver todas las caracteristicas de las claves del diccionario
listas = clientes.values()
print(listas)

# Unir 2 diccionarios
clientes.update(cajero)
print(clientes)