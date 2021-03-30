from Car import Car

car = Car()
print("Press a Key: Forward(W)  Left(A)  Right(D)  Reverse(X)  Stop(S)   Quit(q)")

while(True):
    value = raw_input()      # If you use Python 2
    #value = input()           # If you use Python 3
    #print(value)

    if(value=='q' or value=='Q'):
        car.stopCar()
        car.__exit__()
        exit(0)
    elif(value == 'w' or value == 'w') :
        car.forward()
    elif (value == 'a' or value == 'A'):
        car.turn_left()
    elif (value == 'd' or value == 'D'):
        car.turn_right()
    elif (value == 'x' or value == 'X'):
        car.reverse()
    elif (value == 's' or value == 'S'):
        car.stopCar()


car.__exit__()