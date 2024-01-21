import cv2
import numpy as np;

# Just dummy function for callbacks from trackbar
def display_hsv(x):
    lh = cv2.getTrackbarPos("LH", "Tracking")
    ls = cv2.getTrackbarPos("LS", "Tracking")
    lv = cv2.getTrackbarPos("LV", "Tracking")
    uh = cv2.getTrackbarPos("UH", "Tracking")
    us = cv2.getTrackbarPos("US", "Tracking")
    uv = cv2.getTrackbarPos("UV", "Tracking")
    print((lh, ls, lv), (uh, us, uv))

# Create a trackbar window to adjust the HSV values
# They are preconfigured for a yellow object 
cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 11,255, display_hsv)
cv2.createTrackbar("LS", "Tracking", 151, 255, display_hsv)
cv2.createTrackbar("LV", "Tracking", 157, 255, display_hsv)
cv2.createTrackbar("UH", "Tracking", 25, 255, display_hsv)
cv2.createTrackbar("US", "Tracking", 255, 255, display_hsv)
cv2.createTrackbar("UV", "Tracking", 255, 255, display_hsv)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

# Read test image
cap = cv2.VideoCapture(0)

while True:
    # Convert to HSV colour space
    success, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Read the trackbar values
    lh = cv2.getTrackbarPos("LH", "Tracking")
    ls = cv2.getTrackbarPos("LS", "Tracking")
    lv = cv2.getTrackbarPos("LV", "Tracking")
    uh = cv2.getTrackbarPos("UH", "Tracking")
    us = cv2.getTrackbarPos("US", "Tracking")
    uv = cv2.getTrackbarPos("UV", "Tracking")

    # Create arrays to hold the minimum and maximum HSV values
    hsvMin = np.array([lh, ls, lv])
    hsvMax = np.array([uh, us, uv])
    
    # Apply HSV thresholds 
    mask = cv2.inRange(hsv, hsvMin, hsvMax)
   
    # Uncomment the lines below to see the effect of erode and dilate
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)

    # The output of the inRange() function is black and white
    # so we use it as a mask which we AND with the orignal image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Show the result
    cv2.imshow("Result view", res)

    # Wait for the escape key to be pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()