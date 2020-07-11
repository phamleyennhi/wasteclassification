from picamera import PiCamera
from time import sleep
import shutil
import os

dest_img = '/home/pi/Desktop/wasteclassification/TRAIN/CameraIMGS'
camera = PiCamera()

# get the next image number
img_list = os.listdir(dest_img)
cnt = len(img_list)

src_img = '/home/pi/Desktop/wasteclassification/IMGS/image%s.jpg' %cnt

# capture the image
camera.start_preview()
sleep(5)
camera.capture(src_img)
camera.stop_preview()

# move the image to the training set	
shutil.move(src_img, dest_img)
