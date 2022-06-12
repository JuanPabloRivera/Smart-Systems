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
    
    