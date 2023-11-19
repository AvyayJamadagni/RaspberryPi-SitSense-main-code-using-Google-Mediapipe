import mediapipe as mp
import cv2
import numpy as np
import math
import time
from luma.core.interface.serial import i2c, spi, pcf8574
from luma.core.render import canvas
from luma.oled.device import ssd1306
from PIL import Image
import os

serial = spi(device=0, port=0)
device = ssd1306(serial)

size = 128, 64

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

nyval = [0]
lsholyval = [0]
anglelist = [0]

TNTimeCounter = 0
SlTimeCounter = 0
firstTNTime = 0
firstTNIndex = 0
firstSlTime = 0
firstSlIndex = 0
detectedTNTime = 0
detectedSlTime = 0
TNthresh = 0.05
Slthresh = 0.1
TNtimeThresh = 10/60
SltimeThresh = 15/60
cyclePeriod = 25/60



#Displaying information on OLED
def infoDisplay(stretch):

    with canvas(device) as draw:
        img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'{stretch}.png'))
        img = Image.open(img_path)  
        img = img.resize((128, 64)) \
        .transform(device.size, Image.AFFINE, (1, 0, 0, 0, 1, 0), Image.BILINEAR) \
        .convert(device.mode)
        device.display(img)



#Displaying stretch animations
def stretchDisplay(stretch, animnum):

    with canvas(device) as draw:
        for repeat in range(1,6):
            for picnum in range(1,animnum+1):
                img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'{stretch}{picnum}.png'))
                img = Image.open(img_path)  
                img = img.resize((128, 64)) \
                .transform(device.size, Image.AFFINE, (1, 0, 0, 0, 1, 0), Image.BILINEAR) \
                .convert(device.mode)
                device.display(img)
                time.sleep(1)



#Determining the text neck
def isTextNeck(angle, initangle, anglelist):

    state = False

    if angle <= (1 - TNthresh) * initangle:

        global TNTimeCounter 
        TNTimeCounter += 1

        if TNTimeCounter == 1:

            global firstTNTime 
            firstTNTime = time.time()
            global firstTNIndex 
            firstTNIndex = len(anglelist) - 1

        currentTime = time.time()
        meanAngle = sum(anglelist[firstTNIndex:])/len(anglelist[firstTNIndex:])

        #To check if the text neck has been held for a prolonged time
        if ((currentTime - firstTNTime) >= (TNtimeThresh * 60)) and (meanAngle <= ((1 - TNthresh) * initangle)):
            state = True
            detectedTNTime = time.time()

    return state

#Determining slouch
def isSlouch(nosey, lsholy, initnosey, initlsholy, nyval, lsholyval):

    state = False

    if (nosey >= (1 + Slthresh) * initnosey) and (lsholy >= (1 + Slthresh) * initlsholy):
        global SlTimeCounter
        SlTimeCounter += 1

        if SlTimeCounter == 1:
            global firstSlTime
            firstSlTime = time.time()
            global firstSlIndex
            firstSlIndex = len(nyval) - 1

        currentTime = time.time()

        meanNosey = sum(nyval[firstSlIndex:])/len(nyval[firstSlIndex:])
        meanLsholy = sum(lsholyval[firstSlIndex:])/len(lsholyval[firstSlIndex:])

        #To check if the text neck has been held for a prolonged time
        if ((currentTime - firstSlTime) >= (SltimeThresh * 60)) and ((meanNosey >= (1 + Slthresh) * initnosey) and (meanLsholy >= (1 + Slthresh) * initlsholy)):
            state = True
            detectedSlTime = time.time()

    return state

#Function to calculate angle
def calculateAngle(a,b,c):

    a = np.array(a) #first
    b = np.array(b) #mid
    c = np.array(c) #end

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]-np.arctan2(a[1]-b[1], a[0]-b[0]))
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle



# Phase 1 calculates initial coordinates

def phase1func(initPosRecTim, elapinitPosRecTim):

    print("into phase1func")

    while cap.isOpened():

        ret, frame = cap.read()
        image = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        results = pose.process(image)

        try:

            landmarks = results.pose_landmarks.landmark
            leftShoulder = [landmarks[11].x, landmarks[11].y, landmarks[11].z]
            nose = [landmarks[0].x, landmarks[0].y, landmarks[0].z]
            leftEar = [landmarks[7].x, landmarks[7].y, landmarks[7].z]

            angle = calculateAngle(leftShoulder, nose, leftEar)
            nyval.append(landmarks[0].y)
            lsholyval.append(leftShoulder[1])
            anglelist.append(angle)
            elapinitPosRecTim = time.time() #resetting the elapsed timer

        except:
            pass

        if (cv2.waitKey(10) & 0xFF == ord('q')) or elapinitPosRecTim - initPosRecTim > 10:
            break

    initnyval = sum(nyval[1:])/len(nyval[1:])
    initlsholyval = sum(lsholyval[1:])/len(lsholyval[1:])
    initangle = sum(anglelist[1:])/len(anglelist[1:])

    return initnyval, initlsholyval, initangle

def phase2func(initialTime, initnyval, initlsholyval, initangle):
    print("into phase2func")

    while cap.isOpened():

        ret, frame = cap.read()
        image = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        results = pose.process(image)

        try:
            landmarks = results.pose_landmarks.landmark

            leftShoulder = [landmarks[11].x, landmarks[11].y, landmarks[11].z]
            nose = [landmarks[0].x, landmarks[0].y, landmarks[0].z]
            leftEar = [landmarks[7].x, landmarks[7].y, landmarks[7].z]

            angle = calculateAngle(leftShoulder, nose, leftEar)
            nyval.append(landmarks[0].y)
            lsholyval.append(leftShoulder[1])
            anglelist.append(angle)

            #To check for both text neck and slouch
            if isTextNeck(angle, initangle, anglelist) and isSlouch(landmarks[0].y, leftShoulder[1], initnyval, initlsholyval, nyval, lsholyval):
                #Code to do stretches goes here
                stretchDisplay("handsside", 2)
                break

            #To check for text neck
            elif isTextNeck(angle, initangle, anglelist):
                #Code to do stretches goes here
                if firstSlTime != 0:
                    stretchDisplay("handsside", 2)
                else:
                    stretchDisplay("handsback", 2)
                break

            #To check for slouch
            elif isSlouch(landmarks[0].y, leftShoulder[1], initnyval, initlsholyval, nyval, lsholyval):
                #Code to do stretches goes here
                if firstTNTime != 0:
                    stretchDisplay("handsside", 2)
                else:
                    stretchDisplay("hands", 2)
                break

            currentTime = time.time()

            #To check if the cycle is over
            if (currentTime - initialTime) >= cyclePeriod * 60:
                #User has sat in good posture for 25 min and now needs to be rewarded and should move
                infoDisplay("thumbsup")
                time.sleep(5)
                break

        except:
            pass
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

#Final run through function
def runThrough():

    stretchDisplay("starting", 2)
    initialTime = time.time()
    elapsedTime = time.time()
    initnyval, initlsholyval, initangle = phase1func(initialTime, elapsedTime)
    print(initnyval, initlsholyval, initangle)
    initialTime = time.time()
    phase2func(initialTime, initnyval, initlsholyval, initangle)
    print("finished phase2func")
    time.sleep(5)
    print("resetting variables")   

    global nyval
    nyval = [0]   

    global lsholyval
    lsholyval = [0]

    global anglelist
    anglelist = [0]   

    global TNTimeCounter
    TNTimeCounter = 0    

    global SlTimeCounter
    SlTimeCounter = 0    

    global firstTNTime
    firstTNTime = 0   

    global firstTNIndex
    firstTNIndex = 0

    global firstSlTime
    firstSlTime = 0

    global firstSlIndex
    firstSlIndex = 0
    
    global detectedTNTime
    detectedTNTime = 0

    global detectedSlTime
    detectedSlTime = 0

    print("starting again")



cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    runThrough()
    runThrough()

cap.release()
cv2.destroyAllWindows()
