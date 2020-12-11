import cv2, sys
import numpy as np
from os import listdir
from os.path import join

cascadeRightEar = cv2.CascadeClassifier("haarcascade_mcs_rightear.xml")
cascadeLeftEar = cv2.CascadeClassifier("haarcascade_mcs_leftear.xml")

# Return top left and bottom right coordinates
def getContourCoordinates(top, bottom):
	x1 = top[0][0]
	y1 = top[0][1]

	x2 = bottom[0][0]
	y2 = bottom[0][1]
	return x1, y1, x2, y2

# Return real detection rectangles
def getControus(filename):
	im = cv2.imread(join("testannot_rect", filename))
	imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	boxes = []
	for box in contours:
		x1, y1, x2, y2 = getContourCoordinates(box[0], box[2])
		boxes.append([x1, y1, x2, y2])

	return boxes

def getIou(detection, rectangle):
	dx1, dy1, dx2, dy2 = detection
	rx1, ry1, rx2, ry2 = rectangle

	dx = min(dx2, rx2) - max(dx1, rx1)
	dy = min(dy2, ry2) - max(dy1, ry1)

	p_intersection = 0

	if dx >= 0 and dy >= 0:
		p_intersection = dx * dy
	else: 
		return 0
	
	p_d = (dx2 - dx1) * (dy2 - dy1)
	p_r = (rx2 - rx1) * (ry2 - ry1)

	p_overlap = p_intersection / (p_d + p_r - p_intersection)

	return p_overlap


def detectEar(filename):
	tp = 0
	fn = 0
	fp = 0

	image = cv2.imread(join("test", filename))
	leftEar = cascadeLeftEar.detectMultiScale(image, 1.02, 4)
	rightEar = cascadeRightEar.detectMultiScale(image, 1.02, 4)
	earsDetected = [*leftEar, *rightEar]

	rectangles = getControus(filename)

	for x, y, w, h in earsDetected:
		cv2.rectangle(image, (x,y), (x+w, y+h), (128, 255, 0), 2)
		for i in range(len(rectangles)):
			rect = rectangles[i]
			overlap = getIou([x, y, x+w, y+h], rect)

			if overlap >= 40:
				tp = tp + 1
				rectangles.pop(i)

			cv2.rectangle(image, (rect[0],rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)
	
	if len(earsDetected) == 0:
		print("None detected")

	cv2.imshow("slika", image)
	cv2.waitKey()

	print(tp, fn, fp)

tp = 0 #Pravilno zaznani
fn = 0 #Nepravilno zaznani
fp = 0 #Nezaznani

for f in listdir("test"):
	detectEar(f)

#getIou([340, 90, 377, 151], [341, 88, 377, 155])