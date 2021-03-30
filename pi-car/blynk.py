import blynklib
from Car import Car
import time

blynk = blynklib.Blynk('BmtiNBoOCluRwe3oP5-wYvWRZmGV83Sr', server='192.168.5.5', port=8080)
car = Car()
origin = 127

# register handler for virtual pin V4 write event
@blynk.handle_event('write V4')
def write_virtual_pin_handler(pin, value):
    x = int(value[0])
    y = int(value[1])

    print("({0},{1})".format(x, y))

    if (x == origin and y == origin):
        car.stopCar()
    elif (x >= 240 and x <= 254 and y >= 68 and y <= 186):
        car.turn_right()
    elif (x >= 0 and x <= 20 and y >= 58 and y <= 196):
        car.turn_left()
    elif (x >= 68 and x <= 186 and y >= 240 and y <= 254):
        car.forward()
    elif (x >= 58 and x <= 196 and y >= 0 and y <= 20):
        car.reverse()
    else:
        car.stopCar()


###########################################################
# infinite loop that waits for event
###########################################################
while True:
    blynk.run()