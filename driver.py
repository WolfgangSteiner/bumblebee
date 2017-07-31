import RPi.GPIO as GPIO
import os
from time import sleep


pi_blaster = os.open("/dev/pi-blaster", os.O_WRONLY)


def init_gpio():
    GPIO.setmode(GPIO.BOARD)
    for i in (11,12,15,16):
        GPIO.setup(i, GPIO.OUT)

def flip_pins(channel, value):
    GPIO.output(channel, value)
    GPIO.output(channel+1, not value)


def truncate_velocity(vel):
    vel = abs(vel)
    if vel < 0.1:
        vel = 0.0
    elif vel > 1.0:
        vel = 1.0

    return vel


def set_velocity_right(vel):
    flip_pins(11, vel < 0)
    os.write(pi_blaster, "%d=%.3f\n" % (20, truncate_velocity(vel)))


def set_velocity_left(vel):
    flip_pins(15, vel > 0)
    os.write(pi_blaster, "%d=%.3f\n" % (21, truncate_velocity(vel)))


def set_velocities(vel_left, vel_right):
    #if vel_left != 0.0 or vel_right != 0.0:
    #    print "left: ", vel_left, ", right: ", vel_right
    set_velocity_left(vel_left)
    set_velocity_right(vel_right)


def set_velocity_angle(vel, angle):
    if angle > 0:
        left_vel = vel * (1.0 - angle)
        right_vel = vel 
    else:
        left_vel = vel 
        right_vel = vel * (1.0 + angle)
                
    set_velocities(left_vel, right_vel)    


def turn(rate):
    set_velocities(rate, -rate)
    
    

def on_exit():
    set_velocities(0.0, 0.0)

    
import atexit
atexit.register(on_exit)
init_gpio()
    


