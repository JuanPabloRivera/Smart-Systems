import cv2
import numpy as np

class laneDetector:
    def __init__(self, *args, **kwargs):
        pass
        
    def change_color_space(self, color_space):
        self.color_space = color_space
        self.read_threshold_values(color_space)
        
    def read_threshold_values(self, color_space):
        self.threshold_vals = []
        threshold_file = open(f'{color_space}_values', 'r')
        for line in threshold_file.readlines():
            self.threshold_vals.append(np.array([int(x) for x in line.strip().split()]))
        print(self.threshold_vals)
        
    def get_binary_threshold(self, frame):
        # Masking the frame with the threshold values
        mask = cv2.inRange(frame, self.threshold_values[0], self.threshold_values[1])
        masked = cv2.bitwise_and(frame, frame, mask=mask)

        # Getting a thresolded image
        gray = masked[:,:,0]+masked[:,:,1]  #gray = H values + S values
        ret, gray = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

        return gray
    
    def get_roi(self, frame, interest_percentage):
        x_end = frame.shape[1]
        y_end = round(frame.shape[0]*(1-interest_percentage))
        black_region = np.zeros((y_end, x_end))
        frame[0:y_end, 0:x_end] = black_region
        
    def __get_windows_values(frame, x_current, n_windows, window_height, width_margin, min_recenter_pixels, downwards=False, fitting_curve=False):    
        # list of points
        values = dict()
        values['x_points'] = np.zeros(n_windows, dtype=np.int16)
        values['y_points'] = np.zeros(n_windows, dtype=np.int16)
        values['corners'] = []
        values['windows'] = [False for _ in range(n_windows)]

<<<<<<< HEAD
        # Check if we're going upwards or downwards
        windows = range(n_windows)[::-1] if downwards else range(n_windows)

        for window in windows:        
            y_bottom = frame.shape[0] - (window+1)*window_height
            y_top = frame.shape[0] - window*window_height
            y_current = y_top-(y_top-y_bottom)//2
            x_left = x_current - width_margin
            x_right = x_current + width_margin

            # Getting the index of the non zero pixels in the window
            valid_idx = ((self.nonzeroy >= y_bottom) & (self.nonzeroy < y_top) & (self.nonzerox >= x_left) & (self.nonzerox < x_right)).nonzero()[0]

            # If you found > minpix pixels, recenter next window on their mean position
            if len(valid_idx) > min_recenter_pixels:
                x_current = int(np.mean(self.nonzerox[valid_idx]))
                values['windows'][window] = True # Saving the index of the window to use latter for fitting

            # Saving the newly computed points
            values['x_points'][window] = x_current
            values['y_points'][window] = y_current
            values['corners'].append(((x_left, y_bottom),(x_right, y_top)))

        return values
    
    def get_bad_window_limit(windows):
        for i, window in enumerate(windows):
            if window: return i-1
        return len(windows)

    def get_windows_with_histogram(frame, x_current, nwindows=10, min_recenter_pixels=25, width_margin=50):    
        # Window height
        window_height = frame.shape[0]//(2*nwindows)

        # Pixels that are non zero
        self.nonzero = frame.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])

        values = get_windows_values(frame=frame, nonzerox=nonzerox, nonzeroy=nonzeroy, x_current=x_current, n_windows=nwindows, window_height=window_height, width_margin=width_margin, min_recenter_pixels=min_recenter_pixels)
        
        bad_limit = get_bad_window_limit(values['windows'])
        if  -1 < bad_limit < nwindows: # if there's at least 1 bad window and not all of them are bad
            good_nearest = values['x_points'][bad_limit+1] 
            fixed_values = get_windows_values(frame=frame, nonzerox=nonzerox, nonzeroy=nonzeroy, x_current=good_nearest, n_windows=bad_limit+1, window_height=window_height, width_margin=width_margin, min_recenter_pixels=min_recenter_pixels, downwards=True)

            #Assigning our new fixed values 
            values['x_points'][:bad_limit+1] = fixed_values['x_points']
            values['corners'][:bad_limit+1] = fixed_values['corners']

            # Finally re-sweeping the windows through to adjust upper windows
            values = get_windows_values(frame=frame, nonzerox=nonzerox, nonzeroy=nonzeroy, x_current=values['x_points'][0], n_windows=nwindows, window_height=window_height, width_margin=width_margin, min_recenter_pixels=min_recenter_pixels)
        return values['x_points'], values['y_points'], values['corners'], values['windows']

    def fit_poly_points(x_points, y_points, degree=2, x_range=(0,100)):
        fit = np.polyfit(x_points, y_points, degree)

        x_vals = np.arange(x_range[0], x_range[1])

        try:
            y_vals = (fit[0]*x_vals**(3) + fit[1]*x_vals**(2) + fit[2]*x_vals + fit[3]).astype(int)
        except TypeError as e:
            print(e)
            pass

        return x_vals, y_vals
    
    
=======
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
>>>>>>> 753432fa30c867eced6dffed4e434b79dfdc6169
