import os
import xbox
from time import sleep
import perception
import driver


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--auto", help="drive autonomously", action="store_true")
args = parser.parse_args()

    
if __name__ == "__main__":
    #perception.display_frame = True
    
    if args.auto:
        while True:
            angle = 1.5 * perception.perception()
            print "angle: %.2f" % angle
            driver.set_velocity_angle(0.30, angle)
            
    else:
        joy = xbox.Joystick()

        while True:
            vel = joy.leftY()

            if abs(vel) < 0.1:
                driver.set_velocities(joy.rightX(),-joy.rightX())

            else:
                angle = joy.rightX()
                driver.set_velocity_angle(vel, angle)
 
            sleep(0.01)
        
        joy.close()

