clientes={}

print("================================")
print("==== Bienvenido a red locar ====")
print("================================")

print("si es para agregar un cliente     (1)")
print("si es para listar a los clientes  (2)")
print("si es para borrar a un cliente    (3)")
print("si es para editar a un cliente    (4)")


i = 1
while True:
    option = int(input("\n - Ingrese una opcion: "))
    match option:
        case 1:

                print("====================================")
                print("    Datos del cliente numero "+ str(i))
                print("====================================")
                nombre = input("Ingrese el nombre del cliente: ")
                clientes["nombre "+str(i)] = nombre
                cel = input("Ingrese el cel del cliente: ")
                clientes["celular "+str(i)] = cel
                direccion = input("Ingrese la direccion: ")
                clientes["direccion "+str(i)] = direccion
                saldo = input("Ingrese el saldo: ")
                clientes["saldo "+str(i)] = saldo
                print("---------------------------\n")
                i+=i

        case 2:
            print("\n")
            print(clientes)
        case 3:
            print("Ingrese el numero del cliente que desea borrar: ")
            y = int(input())
            del clientes["nombre "+str(y)]
            del clientes["celular "+str(y)]
            del clientes["direccion "+str(y)]
            del clientes["saldo "+str(y)]
            print("\n Su nueva lista es: ")
            print(clientes)
        case 4:
              while True:
                print("Si quiere cambiar el nombre ingrese      (1)")
                print("Si quiere cambiar el celular ingrese     (2)")
                print("Si quiere cambiar la direccion ingrese   (3)")
                print("Si quiere cambiar el saldo ingrese       (4) ")
                print(" O si quiere salir oprima   '0' ")
                option = int(input("\n - Ingrese  "))
                match option:
                    case 1:
                        print("Ingrese el numero del cliente que quiere editar: ")
                        x = int(input())
                        nombre = input("Ingrese el nombre del cliente: ")
                        clientes["nombre "+str(x)] = nombre
                    case 2:
                        print("Ingrese el celular del cliente que quiere editar: ")
                        x = int(input())
                        celular = input("Ingrese el celular del cliente: ")
                        clientes["celular "+str(x)] = celular
                    case 3:
                        print("Ingrese la direccion del cliente que quiere editar: ")
                        x = int(input())
                        direccion = input("Ingrese la direccion del cliente: ")
                        clientes["direccion "+str(x)] = direccion
                    case 4:
                        print("Ingrese el saldo del cliente que quiere editar: ")
                        x = int(input())
                        saldo = input("Ingrese el saldo del cliente: ")
                        clientes["saldo "+str(x)] = saldo
                    case 0:
                        break
              
        case 0:
            print(" Gracias por su registro!!")
            break
        case _:
            print("Valor ingresado no valido, intente nuevamente")






# for i in range(10):
#     if i == 5:
#         clientes["nombre"+str(i)] = "juan"
#         print(clientes)


# for i in range(1,3):
#     nombre = input("Ingrese el nombre del cliente: ")
#     clientes["nombre"+str(i)] = nombre
#     cel = input("Ingrese el cel del cliente: ")
#     clientes["celular"+str(i)] = cel
#     direccion = input("Ingrese la direccion: ")
#     clientes["direccion"+str(i)] = direccion
#     saldo = input("Ingrese el saldo: ")
#     clientes["saldo"+str(i)] = saldo
#     salto = print("------------------\n")
#     clientes["."]=salto
    
    
# print(clientes)

