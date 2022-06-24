import cv2
import numpy as np

class laneDetector:
    @classmethod
    def __init__(self):
        # Threshold values
        self.threshold_vals = (np.array([0,0,0]),np.array([255,255,255]))
        
        # Number of sliding windows
        self.nwindows = 15
        
        # Starting value of the windows
        self.left_windows = [False for _ in range(self.nwindows)]
        self.right_windows = [False for _ in range(self.nwindows)]
        
        # Polynomial fit logic variables
        self.fit_attempt_left = False
        self.fit_attempt_right = False
        self.fitted_left = False
        self.fitted_right = False
        
        # Polynomial fit degree
        self.degree = 3
        
        # Starting polynomial values
        self.prev_left_fit = [0 for _ in range(self.degree+1)]
        self.prev_right_fit = [0 for _ in range(self.degree+1)]
    
    @classmethod
    def read_threshold_values(self, color_space):
        # Reading threshhold values for the mask
        threshold_vals = []
        threshold_file = open(f'{color_space}_values', 'r')
        for line in threshold_file.readlines():
            threshold_vals.append(np.array([int(x) for x in line.strip().split()]))
        self.threshold_vals = threshold_vals
    
    @staticmethod
    def get_binary_threshold(frame, threshold_values):
        # Masking the frame with the threshold values
        mask = cv2.inRange(frame, threshold_values[0], threshold_values[1])
        masked = cv2.bitwise_and(frame, frame, mask=mask)

        # Finding edges/contours
        #edges = cv2.Canny(masked,100,200)

        # Getting a thresolded image
        gray = masked[:,:,0]+masked[:,:,1]  #gray = H values + S values
        ret, gray = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

        return gray
    
    @staticmethod
    def get_roi(frame, interest_percentage):
        x_end = frame.shape[1]
        y_end = round(frame.shape[0]*(1-interest_percentage))
        black_region = np.zeros((y_end, x_end))
        frame[0:y_end, 0:x_end] = black_region
    
    @staticmethod
    def get_windows_values(frame, nonzerox, nonzeroy, x_values, n_windows, window_height, width_margin, min_recenter_pixels, downwards=False):    
        # list of points
        values = dict()

        # Start the next windows where the mean from the previous wnidow is
        follow_previous = False

        # Checking if x values are provided or should be found
        if len(x_values) == 1:
            values['x_points'] = np.zeros(n_windows, dtype=np.int16)
            values['x_points'][0] = x_values[0]
            follow_previous = True

        elif len(x_values) == n_windows:
            values['x_points'] = x_values
        else: raise ValueError('x_values must be either a single point o a list of values for each of the windows')

        values['y_points'] = np.zeros(n_windows, dtype=np.int16)
        values['corners'] = []
        values['windows'] = [False for _ in range(n_windows)]

        # Check if we're going upwards or downwards
        windows = range(n_windows)[::-1] if downwards else range(n_windows)

        for window in windows:        
            y_bottom = frame.shape[0] - (window+1)*window_height
            y_top = frame.shape[0] - window*window_height
            y_current = y_top-(y_top-y_bottom)//2
            x_left = values['x_points'][window] - width_margin
            x_right = values['x_points'][window] + width_margin

            # Getting the index of the non zero pixels in the window
            valid_idx = ((nonzeroy >= y_bottom) & (nonzeroy < y_top) & (nonzerox >= x_left) & (nonzerox < x_right)).nonzero()[0]

            # If you found > minpix pixels, recenter next window on their mean position
            if len(valid_idx) > min_recenter_pixels:
                values['x_points'][window] = int(np.mean(nonzerox[valid_idx]))
                values['windows'][window] = True # Saving the index of the window to use latter for fitting

            if follow_previous and window < n_windows-1: values['x_points'][window+1] = values['x_points'][window]

            # Saving the newly computed points
            values['y_points'][window] = y_current
            values['corners'].append(((x_left, y_bottom),(x_right, y_top)))

        return values
    
    @staticmethod
    def get_bad_window_limit(windows):
        for i, window in enumerate(windows):
            if window: return i-1
        return len(windows)
    
    @staticmethod
    def get_windows_with_histogram(frame, x_current, nwindows=10, min_recenter_pixels=25, width_margin=50):    
        # Window height
        window_height = frame.shape[0]//(2*nwindows)

        # Pixels that are non zero
        nonzero = frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        values = laneDetector.get_windows_values(frame=frame, nonzerox=nonzerox, nonzeroy=nonzeroy, x_values=[x_current], n_windows=nwindows, window_height=window_height, width_margin=width_margin, min_recenter_pixels=min_recenter_pixels)
        bad_limit = laneDetector.get_bad_window_limit(values['windows'])

        if  -1 < bad_limit < nwindows: # if there's at least 1 bad window and not all of them are bad
            good_nearest = values['x_points'][bad_limit+1] 
            fixed_values = laneDetector.get_windows_values(frame=frame, nonzerox=nonzerox, nonzeroy=nonzeroy, x_values=[good_nearest], n_windows=bad_limit+1, window_height=window_height, width_margin=width_margin, min_recenter_pixels=min_recenter_pixels, downwards=True)

            #Assigning our new fixed values 
            values['x_points'][:bad_limit+1] = fixed_values['x_points']
            values['corners'][:bad_limit+1] = fixed_values['corners']

            # Finally re-sweeping the windows through to adjust upper windows
            values = laneDetector.get_windows_values(frame=frame, nonzerox=nonzerox, nonzeroy=nonzeroy, x_values=[values['x_points'][0]], n_windows=nwindows, window_height=window_height, width_margin=width_margin, min_recenter_pixels=min_recenter_pixels)
        return values['x_points'], values['y_points'], values['corners'], values['windows']
    
    @staticmethod
    def get_windows_with_polynomial(frame, fit, nwindows=10, min_recenter_pixels=25, width_margin=50):
        # Window height
        window_height = frame.shape[0]//(2*nwindows)

        # Getting our defined y values to compute the x ones
        x_vals = np.polyval(fit, self.y_points).astype(int)

        # Pixels that are non zero
        nonzero = frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        values = laneDetector.get_windows_values(frame=frame, nonzerox=nonzerox, nonzeroy=nonzeroy, x_values=x_vals, n_windows=nwindows, window_height=window_height, width_margin=width_margin, min_recenter_pixels=min_recenter_pixels)
        return values['x_points'], values['y_points'], values['corners'], values['windows']
    
    @staticmethod
    def fit_poly_points(x_points, y_points, degree=2, x_range=(0,100)):
        fit = np.polyfit(x_points, y_points, degree)
        x_vals = np.arange(x_range[0], x_range[1])

        try:
            y_vals = np.polyval(fit, x_vals)
        except TypeError as e:
            print(e)

        return fit, (x_vals, y_vals)
    
    @classmethod
    def find_reference(self, frame):
        # Creating the frame were we're gonna see the regression and windows
        frame_debug = frame.copy()

        # Changing to different color spaces
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Choosing the frame we are gonna use for the lane detection sliding windows -> gray; hough lines -> edges
        working_frame = laneDetector.get_binary_threshold(hsv_frame, self.threshold_vals)

        # Taking into account only certain percetage of the image (bottom half)
        laneDetector.get_roi(working_frame, 0.5)

        # Getting the window center points for both left and right lane
        # If there is no fitted curve 
        if not self.fitted_left: 
            histogram = np.sum(working_frame[working_frame.shape[0]//2:, :working_frame.shape[1]//2], axis=0) # Half of the frame width
            left_xcurrent = np.argmax(histogram)
            try:
                (left_xpoints, y_points, left_corners, self.left_windows) = laneDetector.get_windows_with_histogram(working_frame, left_xcurrent, nwindows=self.nwindows, min_recenter_pixels=25, width_margin=50) 
            except Exception as e:
                print(e)

        # Else fit the polynomial with the window centers
        else:
            try:
                (left_xpoints, y_points, left_corners, self.left_windows) = laneDetector.get_windows_with_polynomial(working_frame, self.prev_left_fit, nwindows=self.nwindows, min_recenter_pixels=25, width_margin=50)
            except Exception as e:
                print(e)

        # If there are not enough valid windows to fit a polynomial
        if not self.fitted_right:
            histogram = np.sum(working_frame[working_frame.shape[0]//2:, working_frame.shape[1]//2:], axis=0) # Half of the frame width
            right_xcurrent = np.argmax(histogram) + working_frame.shape[1]//2
            
            try:
                (right_xpoints, y_points, right_corners, self.right_windows) = laneDetector.get_windows_with_histogram(working_frame, right_xcurrent, nwindows=self.nwindows, min_recenter_pixels=25, width_margin=50) 
            except Exception as e:
                print(e)

        # Else fit the polynomial
        else:
            try:
                (right_xpoints, y_points, right_corners, self.right_windows) = laneDetector.get_windows_with_polynomial(working_frame, self.prev_right_fit, nwindows=self.nwindows, min_recenter_pixels=25, width_margin=50)
            except Exception as e:
                print(e)

        # Drawing windows and it's centers -------------------------------------------------------------------------------------------------------
        try:
            for i, val in enumerate(self.left_windows):
                color = (0,255,255) if val else (255,0,255)
                cv2.circle(frame_debug, (left_xpoints[i], y_points[i]), 2, color, 2)
                cv2.rectangle(frame_debug, (left_corners[i][0][0], left_corners[i][0][1]), (left_corners[i][1][0], left_corners[i][1][1]), (0,0,255), 2)
        except Exception as e:
            print('left windows center points are yet not defined', e)

        try:
            for i, val in enumerate(self.right_windows):
                color = (0,255,255) if val else (255,0,255)
                cv2.circle(frame_debug, (right_xpoints[i], y_points[i]), 2 , (0,255,255), 2)
                cv2.rectangle(frame_debug, (right_corners[i][0][0], right_corners[i][0][1]), (right_corners[i][1][0], right_corners[i][1][1]), (0,0,255), 2)
        except Exception as e:
            print('right windows center points are yet not defined', e)

        # Fitting a third degree polynomial curve to the points --------------------------------------------------------------------------------------
        # if there are at least degree+1 points to fit a curve then do it
        if sum(self.left_windows) > self.degree:
            print(left_xpoints, self.left_windows)
            good_x_left = left_xpoints[self.left_windows]
            good_y_left = y_points[self.left_windows]

            try:
                self.prev_left_fit, left_fit = laneDetector.fit_poly_points(good_y_left, good_x_left, degree=self.degree, x_range=(working_frame.shape[0]//2, working_frame.shape[0]))
                self.fit_attempt_left = True
                for i in range(len(left_fit[0])):
                    cv2.circle(frame_debug, (int(left_fit[1][i]), int(left_fit[0][i])), 1, (255,255,255), 1)
            except Exception as e:
                print('Not enough points to fit', e) 
        else: self.fit_attempt_left = False

        if sum(self.right_windows) > self.degree:
            good_x_right = right_xpoints[self.right_windows]
            good_y_right = y_points[self.right_windows]
            try: 
                self.prev_right_fit, right_fit = laneDetector.fit_poly_points(good_y_right, good_x_right, degree=self.degree, x_range=(working_frame.shape[0]//2, working_frame.shape[0]))
                self.fit_attempt_right = True
                # Drawing the 3rd degree poly curve
                for i in range(len(right_fit[0])):
                    cv2.circle(frame_debug, (int(right_fit[1][i]), int(right_fit[0][i])), 1, (255,255,255), 1)
            except Exception as e:
                print('Not enough points to fit', e)
        else: self.fit_attempt_right = False

        # Check if the lines are merging
        # If we're trying to fit a left lane and it's not the same as an already fit right lane
        if self.fit_attempt_left:
            if self.fitted_right:
                if all(self.prev_left_fit == self.prev_right_fit): self.fitted_left = False
                else: self.fitted_left = True
            else: self.fitted_left = True
        else: self.fitted_left = False

        # If we're trying to fit a left lane and it's not the same as an already fit right lane
        if self.fit_attempt_right:
            if self.fitted_left:
                if all(self.prev_left_fit == self.prev_right_fit): self.fitted_right = False
                else: self.fitted_right = True
            else: self.fitted_right = True
        else: self.fitted_right = False

        if self.fitted_left and self.fitted_right:
            # Drawing polygon with left and right window center values
            poly_y = np.concatenate((good_y_left, good_y_right[::-1]), axis=None)
            poly_x = np.concatenate((good_x_left, good_x_right[::-1]), axis=None)

            # Format taken by the fillPoly function np.array([[x1,y1],[x2,y2],[x3,y3],...] )
            poly_points = np.array([[x, y] for x, y in zip(poly_x, poly_y)], dtype = np.int32)
            
            # Drawing it a new frame so we later merge them together with certain alpha
            lane_search_area = np.zeros(frame.shape, dtype=np.uint8)
            cv2.fillPoly(lane_search_area, [poly_points], (160,255,160))
            frame = cv2.addWeighted(src1=frame, alpha=0.6, src2=lane_search_area, beta=0.4, gamma = 0)

        # Drawing line in the middle of the screen
        cv2.line(frame_debug, (frame_debug.shape[1]//2, 0), (frame_debug.shape[1]//2, frame_debug.shape[0]), (255,255,255), 1)
        
        #cv2.imshow('frame', frame)
        #cv2.imshow('debug', frame_debug)
        #cv2.imshow('edges', edges)
        #cv2.imshow('gray', working_frame)
        
        return frame, frame_debug
        
if __name__ == '__main__':
    detector = laneDetector()
    detector.read_threshold_values('HSV')
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if frame is None:
            print('No frame')
            break
        
        lanes, debug = detector.find_reference(frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: break
        
        cv2.imshow('lanes', lanes)
        
    cap.release()
    cv2.destroyAllWindows()

