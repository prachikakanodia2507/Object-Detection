import cv2
#from cv2 import displayOverlay
import jetson.inference
import jetson.utils
 
net = jetson.inference.detectNet("ssd-mobilenet-v2",threshold=0.5)
cam= cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
while True:
    sucess,image=cam.read()
    imgCuda = jetson.utils.cudaFromNumpy(image)
    detections = net.Detect(imgCuda)
    image = jetson.utils.cudaToNumpy(imgCuda)
    cv2.imshow("image",image)
    cv2.waitKey(1)
