from piCamera import PiCamera
import cv2
import numpy as np


def position(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        global cX, cY, clicked
        cX, cY = x, y
        clicked = True

camera = PiCamera()

cX = cY = 0
clicked = False

while camera.cap.isOpened():
    ret, frame =  camera.cap.read()

    # Changing to different color spaces
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    #hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Masking with target values
    lower = np.array([0, 110, 80])
    upper = np.array([255, 140, 110])
    mask = cv2.inRange(lab_frame, lower, upper)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Finding edges/contours
    edges = cv2.Canny(masked,100,200)
    
    # Finding line segments
    #lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=20,maxLineGap=1)
    #if lines is not None:
    #    for line in lines:
    #        x1,y1,x2,y2 = line[0]
    #        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
    #else: print('No lines detected')
    
    # Checking if user wants to exit ('q' or esc)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27: break
    
    
    # Binding callback function and printing pixel values at event location
    cv2.namedWindow(winname='Frame')
    cv2.setMouseCallback('Frame', position)
    if clicked:
        #print(f'R:{frame[cY, cX, 2]}, G:{frame[cY, cX, 1]}, B:{frame[cY, cX, 0]}')
        print(f'L:{lab_frame[cY, cX, 0]}, A:{lab_frame[cY, cX, 1]}, B:{lab_frame[cY, cX, 0]}')
        clicked = False
        
    # Assigning resulting frame
    result_frame = masked
        
    # Drawing line in the middle of the screen
    cv2.line(result_frame, (camera.frame_width//2, 0), (camera.frame_width//2, camera.frame_height), (255,255,255), 1)
        
    cv2.imshow('Frame', result_frame)


camera.cap.release()
cv2.destroyAllWindows()
print('Done')