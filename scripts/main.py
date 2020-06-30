# -*- coding: utf-8 -*-
#https://www.youtube.com/watch?v=WQeoO7MI0Bs
#https://www.murtazahassan.com/learn-opencv-3hours/
import cv2
import numpy as np
from matplotlib import pyplot as plt
def empty(a):
    pass

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def get_hist(img):
    hist = np.sum(img[img.shape[0] // 2:, :], axis=0)
    return hist

def extract_mask_only(image_hsv):
    lower=np.array([29,0,0])
    upper=np.array([44,255,255])
    mask=cv2.inRange(image_hsv,lower,upper)
    return mask

def initializeTrackbarswarp_perspective(intialTracbarVals,wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],wT//2, empty)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, empty)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2],wT//2, empty)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, empty)
    cv2.createTrackbar("width", "Trackbars", 0, hT, empty)
    cv2.createTrackbar("height", "Trackbars", 0, hT, empty)

def valTrackbarswarp_perspective(wT=100, hT=200):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    width = cv2.getTrackbarPos("width", "Trackbars")
    height = cv2.getTrackbarPos("height", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
                      (widthBottom , heightBottom ), (wT-widthBottom, heightBottom)])
    return points,width,height

def warpImg (img,points,w,h,inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2,pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    return imgWarp

def drawPoints(img,points):
    for x in range( 0,4):
        cv2.circle(img,(int(points[x][0]),int(points[x][1])),15,(0,0,255),cv2.FILLED)
    return img

font = cv2.FONT_HERSHEY_SIMPLEX


# fontScale
fontScale = 1

# Blue color in BGR
color = (255, 0, 255)

# Line thickness of 2 px
thickness = 2


# cv2.namedWindow("Trackbars")
# cv2.resizeWindow("Trackbars",640,240)
# cv2.createTrackbar("Hue Min","Trackbars",0,179,empty)
# cv2.createTrackbar("Hue Max","Trackbars",179,179,empty)
# cv2.createTrackbar("Sat Min","Trackbars",0,255,empty)
# cv2.createTrackbar("Sat Max","Trackbars",255,255,empty)
# cv2.createTrackbar("Value Min","Trackbars",0,255,empty)
# cv2.createTrackbar("Value Max","Trackbars",255,255,empty)


# intialTracbarVals_warpperspective = [110,208,0,480]
# initializeTrackbarswarp_perspective(intialTracbarVals_warpperspective)
#


# cap=cv2.VideoCapture("/home/rampfire/crop_row_detection/test_images/crop_row_1.jpeg")
while True:
    original_image=cv2.imread("/home/rampfire/crop_row_detection/test_images/crop_row_1.jpeg")
    # success,original_image=cap.read()
    reference_landmarks=original_image.copy()
    contour_drawing_org_image=original_image.copy()
    imgWarpPoints=original_image.copy()

    print(original_image.shape)#rows columns channels
    image_hsv=cv2.cvtColor(original_image,cv2.COLOR_BGR2HSV)
    print("landmark centroid = ",(int(reference_landmarks.shape[1]/2),int(reference_landmarks.shape[0])))
    reference_landmarks = cv2.circle(reference_landmarks, (int(reference_landmarks.shape[1]/2),int(reference_landmarks.shape[0])), 10, (255, 0, 0) , -1)
    reference_landmarks=cv2.line(reference_landmarks, (int(reference_landmarks.shape[1]/2),0), (int(reference_landmarks.shape[1]/2),int(reference_landmarks.shape[0])), (255, 0, 0), 1)


    mask=extract_mask_only(image_hsv)

    image_result=cv2.bitwise_and(original_image,original_image,mask=mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # areas = [cv2.contourArea(c) for c in contours]
    # max_index = np.argmax(areas)
    # cnt=contours[max_index]
    warppoints = np.float32([(72, 38), (original_image.shape[1] - 72, 38),
                         (0, 143), (original_image.shape[1] - 0, 143)])
    # points,wiiiidth,heiiiight = valTrackbarswarp_perspective(original_image.shape[1],original_image.shape[0])
    imgWarp = warpImg(mask, warppoints,original_image.shape[1],original_image.shape[0])

    imgWarpPoints = drawPoints(imgWarpPoints, warppoints)

    closing=imgWarp.copy()
    frame = np.zeros(closing.shape, np.uint8)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    detect_vertical = cv2.morphologyEx(imgWarp, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    needed_contour=[]
    for count_centr,c in enumerate(cnts):
        cv2.drawContours(closing, [c], -1, (36, 255, 12), 2)
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        print(count_centr,cX,cY,M["m00"],M["m10"],M["m01"],"cx cy 00 10 01")
        if abs(cX-int(reference_landmarks.shape[1]/2)) < 30:
            needed_contour.append(c)
            cv2.drawContours(frame, [c], -1, (255, 255, 255), -1)
    # cv2.imshow('Canny Edges After Contouring', edged)
    img_stack=stackImages(0.8,[[original_image,reference_landmarks],[mask,image_result],[imgWarpPoints,imgWarp],[closing,frame]])
    # img_stack=stackImages(1.2,[[imgWarp,frame]])

    # cv2.imshow("original_image",original_image)
    # cv2.imshow("hsv_image",image_hsv)
    # cv2.imshow("mask",mask)
    # cv2.imshow("image_result",image_result)

    cv2.imshow("img_stack",img_stack)
    cv2.waitKey(50000)


# while True:
    # h_min = cv2.getTrackbarPos("Hue Min","Trackbars")
    # h_max = cv2.getTrackbarPos("Hue Max","Trackbars")
    # s_min = cv2.getTrackbarPos("Sat Min","Trackbars")
    # s_max = cv2.getTrackbarPos("Sat Max","Trackbars")
    # v_min = cv2.getTrackbarPos("Value Min","Trackbars")
    # v_max = cv2.getTrackbarPos("Value Max","Trackbars")
    # lower=np.array([h_min,s_min,v_min])
    # upper=np.array([h_max,s_max,v_max])