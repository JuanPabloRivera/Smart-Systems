from mqtt_Client import MQTT_Client
from piDriver import PiDriver
from piCamera import PiCamera
from us_Sensor import US_Sensor
#from laneDetector import LaneDetector
import cv2
import signal
import threading
import time
import numpy as np

class SmartCar:
    def __init__(self):
        # Creating smart car components
        self.client = MQTT_Client()
        self.driver = PiDriver()
        self.camera = PiCamera()
        self.us_sensor = US_Sensor(5,6)
        
        # Initiating lane detector threshold values
        #self.lane_detector = LaneDetector()
        #self.lane_detector.working_color_space = 'HSV'
        #self.lane_detector.update_threshold_values('HSV')
        
        # Defining car params
        self.params = { 'help':False,
                        'debug':False,
                        'capture':False }

    def debug_mode(self):
        self.params['debug'] = True
        self.driver.debug = True
        self.camera.debug = True
        self.us_sensor.debug = False
        #self.lane_detector.debug = True

    def start(self):
        # Initiating MQTT client as well as loop
        self.client.connect()
        self.client.loop_start()
        measurement_count = 0
        # Create deamon thread for reading ultrasonic_sensor
        threading.Thread(target=self.us_sensor.start, daemon=True).start()
        
        try:
            while(self.camera.cap.isOpened()):
                # Checking there's no risk of colission
                dist = min(self.us_sensor.distance, 50)
                measurement_count += 1
                if measurement_count == 5:
                    self.client.publish(topic='PiCarDistance', payload=dist)
                    measurement_count = 0
                
                if dist < 6:
                    if not self.colission:
                        self.driver.stop()
                        self.colission = True
                        self.client.publish(topic='ColissionRisk', payload=self.colission)
                elif dist > 10:
                    self.colission = False
                    self.client.publish(topic='ColissionRisk', payload=self.colission)
                
                # Obtaining frame
                ret, frame = self.camera.cap.read()
                
                # Catching user input
                key = cv2.waitKey(1) & 0xFF

                # Checking to exit program (with 'q' or 'esc')
                if key == ord('q') or key == 27:
                    self.stop()
                    break

                # Otherwise execute user command
                self.__user_command(key)
                if self.params['help']: self.__show_help(frame)
                
                # Finding lanes in the frame
                #self.lane_detector.find_center_reference(frame, interest_percentage=0.5, draw=True)

                # Showing frame
                cv2.imshow('PiCamera', frame)
                
                # Checking to save frame
                if self.params['capture']:
                    self.__save_frame('Assets', frame)
                    self.params['capture'] = False

        except KeyboardInterrupt:
            print("Interrupted by user")
            self.stop()
            
    def stop(self):
        # Stopping motors and sending final msg
        self.__on_x_press()
        
        # Disconnecting from broker
        self.client.loop_stop()
        self.client.disconnect()
        
    def __sync_camera_steer(self):
        # Find out the value of the steering angle (50-140)
        steer_angle = self.driver.params['steer']
        
        # Adjust the camera pan accordingly (135-45)
        if steer_angle == 90:
            camera_angle = 85
        elif steer_angle < 90:
            camera_angle = 135 - (steer_angle-50)*1.2
        else:
            camera_angle = 85 - (steer_angle-90)*0.8
            
        self.camera.change_pan(camera_angle)
        
    def __save_frame(self, path, frame):
        name = '_'.join([str(x) for x in time.localtime()[0:6]])
        cv2.imwrite(f'{path}/{name}.jpg', frame)
        
    def __user_command(self, key):
        # Motor controls
        if key == ord('w') and not self.colission: self.__on_w_press()
        elif key == ord('s'): self.__on_s_press()
        elif key == ord('a'): self.__on_a_press()
        elif key == ord('d'): self.__on_d_press()
        elif key == ord('x'): self.__on_x_press()

        # Camera controls
        elif key == ord('i'): self.camera.increment_tilt(5)
        elif key == ord('k'): self.camera.increment_tilt(-5)
        elif key == ord('j'): self.camera.increment_pan(5)
        elif key == ord('l'): self.camera.increment_pan(-5)
        
        # Help and car data
        elif key == ord('h'): self.params['help'] = not self.params['help']
        
        # Save current frame
        elif key == 32: self.params['capture'] = True
        
    def __on_w_press(self):
        self.driver.increment_speed(10)
        self.client.publish(topic='PiCarSpeed', payload=self.driver.params['speed'])
        
    def __on_s_press(self):
        self.driver.increment_speed(-10)
        self.client.publish(topic='PiCarSpeed', payload=self.driver.params['speed'])
        
    def __on_a_press(self):
        self.driver.increment_steer(-10)
        self.__sync_camera_steer()
        self.client.publish(topic='PiCarSteer', payload=self.driver.params['steer'])
        
    def __on_d_press(self):
        self.driver.increment_steer(10)
        self.__sync_camera_steer()
        self.client.publish(topic='PiCarSteer', payload=self.driver.params['steer'])
        
    def __on_x_press(self):
        self.driver.stop()
        self.client.publish(topic='PiCarSpeed', payload=self.driver.params['speed'])

    def __show_help(self, frame):
        text = ["help: h",
                "emergency break: x",
                "quit: q/esc",
                "speed up/down: w/s",
                "left/right: a/d",
                "camera up/down: i/k",
                "camera left/right: j/l",
                "start or stop recording: c"]

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,20)
        fontScale = 0.5
        fontColor = (0,0,255)
        lineType = 2
        linepos = 0

        for line in text:
            bottomLeftCornerOfText = (10,20+linepos*20)
            cv2.putText(frame, line, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
            linepos = linepos + 1


if __name__ == "__main__":
    try:
        car = SmartCar()
        car.debug_mode()
        car.start()

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        self.camera.cap.release()
    
    
    
    
    
