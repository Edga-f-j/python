import random


num = random.randint(1,9)


print("================================")
print("       Juego de adivinanza      ")

print("En este juego van a tener 5 intentos para adivinar el numero")

for i in range (1,6):

    print("Intento numero ("+str(i)+"), ingrese un numero")
    x = int(input())

    if x > num:
        print("El numero es mayor, tiene que ser mas bajo")
    elif x < num:
        print("El numero es menor, Trate con otro numero mas alto")
    else:
        print("!!! Felicitaciones, Ha ganado el juego !!!")
        break
        

