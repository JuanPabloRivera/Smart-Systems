import cv2
import numpy as np

def on_low_0_thresh_trackbar(val):
    global low_vals
    low_vals[0] = val
    #print('LOW', low_vals)

def on_high_0_thresh_trackbar(val):
    global high_vals
    high_vals[0] = val
    #print('HIGH', high_vals)
    
def on_low_1_thresh_trackbar(val):
    global low_vals
    low_vals[1] = val

def on_high_1_thresh_trackbar(val):
    global high_vals
    high_vals[1] = val
    
def on_low_2_thresh_trackbar(val):
    global low_vals
    low_vals[2] = val
    
def on_high_2_thresh_trackbar(val):
    global high_vals
    high_vals[2] = val
    
# Defining initial values
low_vals = np.array([0, 0, 0])
high_vals = np.array([255, 255, 255])

# Defining maximum values
max_vals = [255,255,255]

# Defining color space
input_channels = 'HSV'

cap = cv2.VideoCapture(2)

width = int(cap.get(3))
height = int(cap.get(4))

# Creating named window
window_name = 'Frame'
cv2.namedWindow(window_name)

# Creating trackbars
for i, channel in enumerate(input_channels):
    cv2.createTrackbar(f'Low {channel}', window_name, low_vals[i], max_vals[i], locals()[f'on_low_{i}_thresh_trackbar'])
    cv2.createTrackbar(f'High {channel}', window_name, high_vals[i], max_vals[i], locals()[f'on_high_{i}_thresh_trackbar'])

if not cap.isOpened():
    raise  RuntimeError("Unable to read camera feed")
    
try:
    while cap.isOpened():
        ret, frame =  cap.read()
        if frame is None:
            print('No frame')
            break
            
        # Cropping the image to remove logitech border
        frame = frame[75:height-75, 15:width-15, :]

        # Color space mapping
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Masking based of user values
        mask = cv2.inRange(hsv_frame, low_vals, high_vals)
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Checking if user wants to exit ('q' or esc)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: break

        cv2.imshow(window_name, masked)
        
    cap.release()
    cv2.destroyAllWindows()
    
    # Writing threshold values to the respective file
    file = open(f'{input_channels}_values', 'w')
    for val in low_vals: file.write(f'{val} ')
    file.write('\n')
    for val in high_vals: file.write(f'{val} ')
    file.close()
        
except Exception as e: 
    print(e)
    cap.release()
    cv2.destroyAllWindows()
    
cap = cv2.VideoCapture(2)

width = int(cap.get(3))
height = int(cap.get(4))

while cap.isOpened():
    ret, frame =  cap.read()
    if frame is None:
        print('No frame')
        break
        
    # Cropping the image to remove logitech border
    frame = frame[75:height-75, 15:width-15, :]
        
    # Changing to different color spaces
    #lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Masking with target values
    mask = cv2.inRange(hsv_frame, threshold_vals[0], threshold_vals[1])
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Finding edges/contours
    edges = cv2.Canny(masked,100,200)
    
    # Taking into account only certain percetage of the image (bottom half)
    desired_percentage = 0.6
    x_end = edges.shape[1]
    y_end = round(edges.shape[0]*(1-desired_percentage))
    black_region = np.zeros((y_end, x_end))
    edges[0:y_end, 0:x_end] = black_region
    
    # Merging track lines to the erode them into 1
    #edges = cv2.dilate(edges, np.ones(shape = (5,5)), iterations=10)
    #edges = cv2.erode(edges, np.ones(shape = (5,5)), iterations=10)
    
    # Finding line segments
    lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength=50,maxLineGap=10)
    if lines is not None:
        #print(len(lines))
        # Finding the longest line segment on both sides of the screen
        longest_left = [0, (0,0,0,0)] # Distance, (x1,y1,x2,y2)
        longest_right = [0, (0,0,0,0)] # Distance, (x1,y1,x2,y2)
        for line in lines:
            # Calculating the total line distance and saving the longest line
            x1,y1,x2,y2 = line[0]
            
            # Checking how vertical the line is, if it's horizontal, then skip
            angle = get_angle((x1,y1), (x2,y2), (0,0), (100,0), degrees=True)
            
            #if angle < 40:
            cv2.line(frame, (x1,y1), (x2,y2), (255,255,0), 2)
            cv2.putText(frame, f'{round(angle,1)}', org=(x1,y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(75,60,100), thickness=1, lineType=cv2.LINE_AA)
            
            dist = ((y2-y1)**2 + (x2-x1)**2)**0.5
            
            # Checking if line is in the left side
            if x1 <= edges.shape[1]//2 and x2 <= edges.shape[1]//2:
                if dist >= longest_left[0]:
                    longest_left[0] = dist
                    longest_left[1] = line[0]
            # Else in the right side
            else:
                if dist >= longest_right[0]:
                    longest_right[0] = dist
                    longest_right[1] = line[0]
                    
        # Getting the angle between both lines
        #angle = get_angle(longest_left[1][0:2], longest_left[1][2:4], longest_right[1][0:2], longest_right[1][2:4], degrees=True)
        #print(angle)        
            
        # Drawing only the longest line on both sides
        #print(f'first: ({longest_left[1][0]},{longest_left[1][1]})', f'second: ({longest_left[1][2]},{longest_left[1][3]})')
        cv2.line(frame, (longest_left[1][0],longest_left[1][1]), (longest_left[1][2],longest_left[1][3]), (0,255,0), 2)
        cv2.line(frame, (longest_right[1][0],longest_right[1][1]), (longest_right[1][2], longest_right[1][3]), (0,255,0), 2)
        
        # - - - - - Technique No. 1 to find center reference - - - - - #
        # Finding and drawing the middle point on both lines (longest segments)
        middle_left = ((longest_left[1][0]+longest_left[1][2])//2, (longest_left[1][1]+longest_left[1][3])//2)
        middle_right = ((longest_right[1][0]+longest_right[1][2])//2, (longest_right[1][1]+longest_right[1][3])//2)
        cv2.circle(frame, middle_left, 5, (0,0,255), 5)
        cv2.circle(frame, middle_right, 5, (0,0,255), 5)
        # Lines between both points and middle point -> reference to follow
        center_reference = ((middle_left[0]+middle_right[0])//2, (middle_left[1]+middle_right[1])//2)
        cv2.line(frame, middle_left, middle_right, (255,0,0), 2)
        cv2.circle(frame, center_reference, 5, (255,255,255), 5)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
        # - - - - - Technique No. 2 to find center reference - - - - - #
        # Lines between opposite sides of the longest segments
        # Checking which points are up or down
        if longest_left[1][1] > longest_left[1][3]:
            lower_left = (longest_left[1][0], longest_left[1][1])
            upper_left = (longest_left[1][2], longest_left[1][3])
        else: 
            lower_left = (longest_left[1][2], longest_left[1][3])
            upper_left = (longest_left[1][0], longest_left[1][1])
            
        if longest_right[1][1] > longest_right[1][3]:
            lower_right = (longest_right[1][0], longest_right[1][1])
            upper_right = (longest_right[1][2], longest_right[1][3])
        else: 
            lower_right = (longest_right[1][2], longest_right[1][3])
            upper_right = (longest_right[1][0], longest_right[1][1])
        
        # Drawing opposite lines
        cv2.line(frame, upper_left, lower_right, (0,255,255), 2)
        cv2.line(frame, lower_left, upper_right, (0,255,255), 2)
        #Drawing intersection point between both lines
        intersection_reference = find_intersection(upper_left, lower_right, lower_left, upper_right)
        if intersection_reference:
            cv2.circle(frame, intersection_reference, 5, (255,0,255), 5)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    else: pass #print('No lines detected')
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27: break
    
    # Drawing line in the middle of the screen
    cv2.line(frame, (width//2, 0), (width//2, height), (255,255,255), 1)
    
    cv2.imshow('frame', frame)
    cv2.imshow('edges', edges)
    
cap.release()
cv2.destroyAllWindows()