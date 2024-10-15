
########## NUMEROS PARES##########

# numeros_pares = 10
# contador = 5

# def pares(n):
#     print(f'Numeros pares desde 0 hasta {n}: ')
#     for i in range(3, n+1):
#         if i % 2 == 0:
#             print(i)
    
# pares(10)

############# TIENDA ############

# cantidad = int(input('ingrese la cantidad del producto: '))
# valor = int(input('Ingrese el valor del producto: '))

# def total(x, y):
#     return(x*y)
    


# resultado= total(cantidad, valor)

# print(f'El total a pagar es:  ${resultado}')


############ tienda pro ############
# def multiplicacion(x,y):
#     return x*y
# cantidad = 1
# subtotal = 0
# # while cantidad !=0:
# #     cantidad = int(input("Ingrese la cantidad del producto: "))
# #     valor = int(input("Ingrese el valor del producto: $"))
# #     calculo = multiplicacion(valor, cantidad)
# #     subtotal = subtotal + calculo


# rango = int(input("ingrese la cantidad de productos a registrar: "))
# for i in range(rango):
#     cantidad = int(input("Ingrese la cantidad del producto: "))
#     valor = int(input("Ingrese el valor del producto: $ "))
#     calculo = multiplicacion(valor, cantidad)
#     subtotal = subtotal + calculo
#     if i == rango:
#         break

# print(f'El total a pagar es {subtotal}')


nombre =[ ]
celular = []
direccion = []
saldo = []

print("Bienvenido a Banco Caja pobre, seleccione la accion que quiere realizar")
print("si es para agregar un cliente     (1)")
print("si es para listar a los clientes  (2)")
print("si es para borrar a un cliente    (3)")
print("si es para editar a un cliente    (4)")

# seleccion = int(input("Cual accion desea realizar: "))
# clientes = 1
# i = 0

# while clientes != 0:
#     if seleccion == 1:
#         #print("opcion 1")
#         print("Ingrese su nombre completo")
#         nombre.append(input())
#         print("Ingrese su numero de celular")
#         celular.append(input())
#         print("Ingrese su direccion")
#         direccion.append(input())
#         print("Ingrese su saldo")
#         saldo.append(input())
#     elif seleccion == 2:
#         print("Lista de clientes:")
#         print(f"nombre: {nombre[i]}, celular: {celular[i]}, direccion: {direccion[i]}, saldo: {saldo[i]}")
#     elif seleccion == 3:
#         print("opcion 3")
#     elif seleccion == 4:
#         print("opcion 4")
#     else:
#         print("Su opcion es erronea, LEA BIEN!!")
#         break
#     print("Si quiere terminar oprima '0', de lo contrario oprima cualquier numero: ")
#     clientes = int(input())



while True:
    option =  int(input("ingrese la opcion"))
    match option:
        case 1:
            num = int(input("Ingrese el numero de personas a registrar: "))
            for i in range(num):
                print("Ingrese su nombre completo")
                nombre.append(input())
                print("Ingrese su numero de celular")
                celular.append(input())
                print("Ingrese su direccion")
                direccion.append(input())
                print("Ingrese su saldo")
                saldo.append(input())
        case 2:
            num = int(input("Hasta que valor desea ver el arreglo: "))
            for i in range(num):
                print("Lista de clientes:")
                print(f"nombre: {nombre[i]}, celular: {celular[i]}, direccion: {direccion[i]}, saldo: {saldo[i]}")
    

        case 3:
            print("2")
        case 4:
            print("2")
        case 0:
            print("0")
            break
        case _:
            print("no valido")

