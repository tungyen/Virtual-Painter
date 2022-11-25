import cv2
import time
import numpy as np
import os
import HandTrackingModule as htm

#########################
brushThickness = 15
eraserThickness = 50
#########################

folderPath = "Photo"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
	image = cv2.imread(f'{folderPath}/{imPath}')
	overlayList.append(image)

header = overlayList[0]
drawColor = (255,0,255)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon = 0.85)

imgCanvas = np.zeros((720,1280,3),np.uint8)

is_reset = False

while True:

	# 1. Import Image
	success, img = cap.read()
	img = cv2.flip(img, 1)

	# 2. Find Hand Landmarks
	img = detector.findHands(img)
	lmList, _ = detector.findPosition(img, draw=False)

	if len(lmList) != 0:
		#print(lmList)

		# tip of index and middle fingers
		x1, y1 = lmList[8][1:]
		x2, y2 = lmList[12][1:]

		# 3. Check which fingers are up
		fingers = detector.fingersUp()
		#print(fingers)

		# 4. If Selection mode - Two finger are up
		if fingers == [0,1,1,0,0]:
			xp, yp = 0, 0
			is_reset = True
			cv2.rectangle(img, (x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)
			print("Selection Mode")
			# Checking for the Click
			if y1 < 125:
				if 380 < x1 < 480:
					header = overlayList[0]
					drawColor = (255,0,255)
				elif 590 < x1 < 690:
					header = overlayList[1]
					drawColor = (255,0,0)
				elif 800 < x1 < 900:
					header = overlayList[2]
					drawColor = (0,255,0)
				elif 1000 < x1 < 1200:
					header = overlayList[3]
					drawColor = (0,0,0)


		# 5. If Drawing Mode - Index finger is up
		if fingers == [0,1,0,0,0] and is_reset:
			cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
			print("Drawing Mode")
			if xp == 0 and yp == 0:
				xp, yp = x1, y1

			if drawColor == (0,0,0):
				cv2.line(img, (xp, yp), (x1,y1),drawColor,eraserThickness)
				cv2.line(imgCanvas, (xp, yp), (x1,y1),drawColor,eraserThickness)

			else:
				cv2.line(img, (xp, yp), (x1,y1),drawColor,brushThickness)
				cv2.line(imgCanvas, (xp, yp), (x1,y1),drawColor,brushThickness)
			xp,yp = x1,y1

		imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
		_, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
		imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)

		# Since imgInv show the trajectory with black, which is 0 in img
		# the bitwise_and will make img black along with the trajectory
		img = cv2.bitwise_and(img,imgInv)

		# Since the imgCanvas show the trajectory with color, which is 1 in img
		# imgCanvas will make the black route in img color again
		img = cv2.bitwise_or(img,imgCanvas)

	# Setting the header image
	img[0:125,0:1280] = header
	cv2.imshow("Image", img)
	cv2.waitKey(1)
