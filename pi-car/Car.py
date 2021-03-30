import RPi.GPIO as GPIO
import time

class Car():
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(17, GPIO.OUT)
        GPIO.setup(22, GPIO.OUT)
        GPIO.setup(23, GPIO.OUT)
        GPIO.setup(24, GPIO.OUT)
        print("Created Car Object. Ready To Accept Controls!")

    def __exit__(self, exc_type, exc_value, traceback):
        GPIO.cleanup()
        print("Car Object Cleaned!!")

    def stopCar(self):
        print("Stopping Car...")
        GPIO.output(17, False)
        GPIO.output(22, False)
        GPIO.output(23, False)
        GPIO.output(24, False)
        print("Car Stopped!!")

    def forward(self):
        print("Moving Forward...")
        GPIO.output(17, True)
        GPIO.output(22, False)
        GPIO.output(23, True)
        GPIO.output(24, False)

    def reverse(self):
        print("Moving Reverse...")
        GPIO.output(17, False)
        GPIO.output(22, True)
        GPIO.output(23, False)
        GPIO.output(24, True)

    def turn_right(self):
        print("Moving Right...")
        GPIO.output(17, False)
        GPIO.output(22, True)
        GPIO.output(23, True)
        GPIO.output(24, False)

    def turn_left(self):
        print("Moving Left...")
        GPIO.output(17, True)
        GPIO.output(22, False)
        GPIO.output(23, False)
        GPIO.output(24, True)