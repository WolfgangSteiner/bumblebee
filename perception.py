import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep
from threading import Thread
import numpy as np
import angle_prediction

import datetime
#camera = PiCamera()

def file_name(label):
    return "IMG/" + datetime.datetime.now().isoformat().replace("-", "").replace(":", "").replace(".", "") + "_" + label + ".png"


sleep(0.1)

class PiVideoStream:
    def __init__(self, resolution=(160, 120), framerate=8):
        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True)
        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False
        self.is_threaded = False

        sleep(2.0)
        
        self.camera.iso = 800
        print "exposure: ", self.camera.exposure_speed
        self.camera.shutter_speed = 68000 #self.camera.exposure_speed
        self.camera.exposure_mode = 'off'
        g = self.camera.awb_gains
        self.camera.awb_mode = 'off'
        self.camera.awb_gains = g


    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        self.is_threaded = True
        return self

    
    def update(self):
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.frame = f.array
            self.rawCapture.truncate(0)
                
            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.is_threaded = False
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return

    def read(self):
        # return the frame most recently read
        if self.is_threaded:
            return self.frame
        else:
            print "exposure: ", self.camera.exposure_speed
            self.frame = next(self.stream).array
            self.rawCapture.truncate(0)
            return self.frame
        

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

display_frame = False
vs = PiVideoStream().start()
sleep(2.0)

key_table = {65361:"l", 130897:"L", 65363:"r", 130899:"R", 65362:"S"}


def perception():
    delta_x = 120
    orig_frame = vs.read()
    thres = 128
    frame = cv2.inRange(orig_frame, (thres,thres,thres),(255,255,255))
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

#    y_pos, x_pos = frame[:,:].nonzero()
#    left_idx = np.where(x_pos < delta_x)[0] if len(x_pos) else []
#    right_idx = np.where(x_pos >= 320 - delta_x)[0] if len(x_pos) else []

    if frame is not None and display_frame:
        cv2.imshow("Image", orig_frame)
        cv2.imshow("Processed", frame)
        key = cv2.waitKey(1)
        if key in key_table:
            label = key_table[key]
            cv2.imwrite(file_name(label), orig_frame)
            print file_name(label)

    #frame = orig_frame[np.where(frame > 0)]
    angle_prediction.process_img_and_show(orig_frame)
    angle = angle_prediction.predict_angle(orig_frame)
    print angle
    return angle

#    if len(left_idx):
#        left_mean = np.mean(x_pos[left_idx]) 
#    else:
#        left_mean = 0.0

#    if len(right_idx):
#        right_mean = 320 - np.mean(x_pos[right_idx])
#    else:
#        right_mean = 0

#    if len(left_idx) < 10:
#        left_mean = 0

#    if len(right_idx) < 10:
#        right_mean = 0
        
 

#    if left_mean > 0 and right_mean == 0.0:
#        return -0.25
#    elif left_mean == 0 and right_mean > 0:
#        return 0.25
#    elif left_mean == 0 and right_mean == 0:
#        return 0
#    else:
#        diff = left_mean - right_mean       

#        if abs(diff) < 5:
#            return 0.0    
#        elif diff > 0:
#            return -0.25
#        else:
#            return 0.25





if __name__ == "__main__":
    display_frame = True
    
    while True:
	angle = 2.0 * perception()
#	print "angle: %.3f" % angle


    #print "left_mean = %.2f, right_mean = %.2f" % (left_mean, right_mean)



