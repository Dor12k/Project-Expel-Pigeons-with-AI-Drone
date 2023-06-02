
import cv2

import numpy as np

# from time import sleep
# from djitellopy import Tello
from collections import deque
from timeit import default_timer as timer

import tensorflow as tf


# Class represent hte drone camera
class Camera:

    # Load our model that trained by 25 epochs on CIFAR dataset
    MODEL = tf.keras.models.load_model("cifar10_ResNet50_Fine_Tuning_95.h5")

    # Define the labels of CIFAR-10 dataset
    LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Catch frame from webcam
    camera = cv2.VideoCapture(r'C:\WorkSpace\JupyterNotebook\Base\Videos\Birds\Birds 6.mp4')

    # Define variables for height and width shape to prediction model input
    model_width, model_height = 224, 224

    # Define FPS Variable
    FPS = 0

    # Increasing FPS counter
    counter_fps = 0

    # Scale to decrease the frame size
    SCALE = 4

    # Define frame size
    # WIDTH, HEIGHT = 400, 250

    # Define objects boundaries size
    MIN_OBJECT_AREA = 50
    MAX_OBJECT_AREA = 1000

    # Define the background
    # last_frame = np.zeros((int(HEIGHT / SCALE), int(WIDTH / SCALE), 3), np.uint8)
    # background = np.zeros((int(HEIGHT / SCALE), int(WIDTH / SCALE), 3), np.uint8)

    # Restart timer for FPS
    fps_start = timer()

    # Variable store the system status of tracking or not tracking
    # tracking_on = False

    # Define timer to check the tracking
    tracking_check = 10

    # Define the trash hold of the masks
    DIFF_TRASH_HOLD = 20  # Should be low
    MASK_TRASH_HOLD = 100  # Should be high

    # Create background of the main frame
    foregroundModel = cv2.createBackgroundSubtractorMOG2()

    # Define tracker dictionary
    tracker_dict = {'csrt': cv2.legacy.TrackerCSRT_create,
                    'kcf': cv2.legacy.TrackerKCF_create,
                    'mil': cv2.legacy.TrackerMIL_create,
                    'tld': cv2.legacy.TrackerTLD_create(),
                    'medianflow': cv2.legacy.TrackerMedianFlow_create(),
                    'boosting': cv2.legacy.TrackerMOSSE_create(),
                    'mosse': cv2.legacy.TrackerMOSSE_create()}

    def __init__(self, drone, drone_camera):

        self.status = 0
        self.target = 0
        self.patrol = 0

        # Turn drone camera on and off
        self.camera = False

        # Switch test mode
        self.on_test = False

        # Switch drone tracking on and off
        self.tracking = False

        # Set tracking mode on and off
        self.tracking_mode = False

        # Define drone object
        self.drone = drone
        self.drone_frame = drone_camera
        self.drone_last_frame = drone_camera

        # Set width and height according the Process class
        self.width = drone_camera.shape[1]
        self.height = drone_camera.shape[0]

        # Define reading frames counter
        self.counter_frames_reading = 0

        # Define tracking frames counter
        self.counter_frames_tracking = 0

        # Define processing frames counter
        self.counter_frames_processing = 0

        # Define prediction frames counter
        self.counter_frames_prediction = 0

        # Initialize our tracker after the object
        self.tracker = Camera.tracker_dict['csrt']()

        print("Camera class constractor: ", "Camera is : ", self.camera, "You can change test mode by press 'c' ")
        print("Camera class constractor: ", "Test mode is : ", self.on_test, "You can change test mode by press 'k' ")
        print("Camera class constractor: ", "Tracking mode is : ", self.tracking_mode, "You can change test mode by press 'l' ")

    # Connect to drone camera
    def stream_on(self):
        self.drone.drone.streamon()

    # Disconnect from drone camera
    def stream_off(self):
        self.drone.drone.streamoff()

    # Return drome from drone camera
    def read_frame_from_camera(self):

        if self.on_test:
            drone_frame = np.zeros((self.height, self.width, 3), np.uint8)
            drone_mask = np.zeros((self.height, self.width, 1), np.uint8)

            # Return two frames
            return drone_frame, drone_mask
        else:
            # If user request camera then start read and precess frames. else return black screen.
            if self.camera:
                # Function return a frame from drone camera and the mask
                drone_frame, drone_mask_scaled = self.read_frame()
            else:
                # Capturing frames one-by-one from camera
                ret, drone_camera_frame = Camera.camera.read()

                # If the frame was not retrieved then we break the loop
                if not ret or drone_camera_frame is None:
                    drone_frame = np.zeros((self.height, self.width, 3), np.uint8)
                    drone_mask = np.zeros((self.height, self.width, 1), np.uint8)

                    # Return two frames
                    return drone_frame, drone_mask
                else:
                    # Function return processed frame from drone camera and mask
                    drone_frame, drone_mask_scaled, drone_last_frame = Camera.process_frame(self, drone_camera_frame, self.drone_last_frame)

                    # Update the drone camera frame variable
                    self.drone_frame, self.drone_last_frame = drone_frame,  drone_last_frame

            # Resize the mask back to original size
            drone_mask = cv2.resize(drone_mask_scaled, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

            # Expand mask frame to 3 dimension
            # drone_mask = cv2.cvtColor(drone_mask, cv2.COLOR_GRAY2RGB)

            # Return two frames
            return drone_frame, drone_mask

    # Function read frames from drone camera
    def read_frame(self):

        # Counting how many frames we are reading
        self.counter_frames_reading += 1

        # Get a frame from drone camera by the drone SDK
        drone_frame = self.drone.drone.get_frame_read().frame

        # Function return processed frame from drone camera and mask
        drone_frame, drone_mask_scaled, drone_last_frame = Camera.process_frame(self, drone_frame, self.drone_last_frame)

        # Update the drone camera frame variable and the last frame variable
        self.drone_frame, self.drone_last_frame = drone_frame, drone_last_frame

        return drone_frame, drone_mask_scaled

    # Function return processed frame from drone camera, mask, and last frame
    def process_frame(self, drone_camera_frame, last_frame):

        # If tracking mode is on then tracking else just read frames
        if self.tracking_mode:

            # Function return processed frame from drone camera, mask, and last frame in size (WIDTH / SCALE)
            drone_frame, drone_last_frame, drone_frame_scaled, drone_mask_scaled, = Camera.preprocess_frames(self, drone_camera_frame, last_frame)

            if not self.tracking:
                # Function return drone_frame with bounding box
                drone_frame = Camera.detection_manager(self, drone_frame, drone_frame_scaled, drone_mask_scaled)
            else:
                # Function return bounding box of the object we track after it
                drone_frame = Camera.tracking_manager(self, drone_frame, drone_frame_scaled, drone_mask_scaled)

            return drone_frame, drone_mask_scaled, drone_last_frame
        else:
            # Resize the main drone frame to be fit to Process class constant
            self.drone_frame = cv2.resize(drone_camera_frame, (self.width, self.height))

            return self.drone_frame, self.drone_frame, self.drone_frame

    # Function create 3 frames from the frame we read
    def preprocess_frames(self, drone_camera_frame, drone_last_frame):

        # Counting how many frames we're processing
        self.counter_frames_processing += 1

        # Define small sizes
        height, width = int(self.height / Camera.SCALE), int(self.width / Camera.SCALE)

        # Resize the main drone frame to be fit to Process class constant
        self.drone_frame = cv2.resize(drone_camera_frame, (self.width, self.height))

        # Resize the frames to scale shape
        drone_frame_scaled = cv2.resize(self.drone_frame, (width, height))

        drone_last_frame = cv2.resize(self.drone_last_frame, (width, height))

        # Return mask frame
        # drone_mask = Camera.mask(drone_frame_scaled, drone_last_frame)
        drone_mask = Camera.cv_mask(drone_frame_scaled, drone_last_frame)

        # Define last frame
        self.drone_last_frame = self.drone_frame.copy()

        return self.drone_frame, self.drone_last_frame, drone_frame_scaled, drone_mask,

    # Function define the mask
    @staticmethod
    def mask(drone_frame, drone_last_frame):

        # Converting captured frame to GRAY by OpenCV function
        frame_gray = cv2.cvtColor(drone_frame, cv2.COLOR_RGB2GRAY)

        # Create one more frame with Gaussian blur
        frame_gray = cv2.GaussianBlur(frame_gray, (25, 25), 0)

        # Converting captured frame to GRAY by OpenCV function
        drone_last_frame = cv2.cvtColor(drone_last_frame, cv2.COLOR_BGR2GRAY)

        # Create one more frame with Gaussian blur
        drone_last_frame = cv2.GaussianBlur(drone_last_frame, (25, 25), 0)

        # Return mask to detect change between two frames
        abs_diff = cv2.absdiff(frame_gray, drone_last_frame)

        # Function exclude values that ara more than threshold = 15 0 and more than 255
        _, drone_mask = cv2.threshold(abs_diff, 20, 255, cv2.THRESH_BINARY)

        # Dilates the object in the frame
        dilated_mask = cv2.dilate(drone_mask, None, iterations=5)

        return dilated_mask

    # Function create a mask with connectedComponents
    @staticmethod
    def cv_mask(frame_rgb, last_frame):

        # Apply the frame to foreground model
        foreground_mask = Camera.foregroundModel.apply(frame_rgb)

        # Reduce noises
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        foreground_mask = cv2.morphologyEx(np.float32(foreground_mask), cv2.MORPH_OPEN, structuring_element)

        # Find out connected components and keep only the large components
        num_labels, image_labels = cv2.connectedComponents(np.array(0 < foreground_mask, np.uint8))

        # Return components larger than threshold
        foreground_mask = Camera.keep_large_components(image_labels, threshold=0)

        # Using 'clip' function to exclude values that are less than 0 and more than 255
        foreground_mask = np.clip(foreground_mask, 0, 255).astype(np.uint8)

        # Function exclude values that ara more than threshold = 15 0 and more than 255
        _, foreground_mask = cv2.threshold(foreground_mask, 0, 255, cv2.THRESH_BINARY)

        # Converting output feature map from Tensor to Numpy array
        foreground_mask = foreground_mask[:, :, np.newaxis]

        return foreground_mask

    # This function remove the components that are smaller than particular threshold
    @staticmethod
    def keep_large_components(image, threshold):

        frame = np.zeros(image.shape) < 0  # boolean array
        unique_labels = np.unique(image.flatten())  # find out every unique value that is actually a label

        for label in unique_labels:
            if label == 0:  # background
                pass
            else:
                img = (image == label)  # save the component
                if threshold < np.sum(img):
                    frame = frame | img  # save all the components

        return np.float32(255*frame)

    # Method manage the detection object
    def detection_manager(self, drone_frame, drone_frame_scaled, drone_mask_scaled):

        # Counting how many frames we are prediction
        self.counter_frames_prediction += 1

        # Function return array of all contours we found
        contours, _ = cv2.findContours(drone_mask_scaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sorted the contours and define the larger first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Scan the contours list
        for contour in contours:

            # Return square area of the given contour
            contour_area = cv2.contourArea(contour)

            # Find contours between MIN_OBJECT_AREA to MAX_OBJECT_AREA
            if contour_area < Camera.MAX_OBJECT_AREA:
                if Camera.MIN_OBJECT_AREA < contour_area:

                    # Get an approximate rectangle coordinates
                    (x_min, y_min, box_width, box_height) = cv2.boundingRect(contour)

                    # Store the rectangle coordinates around the object
                    rectangle_scaled = np.array([x_min, y_min, box_width, box_height])

                    # Predict image and get the right label
                    label = Camera.prediction_manager(self, drone_frame, rectangle_scaled)

                    # if bird is detected we start the tracking
                    if label == "bird":

                        # Change tracking status on
                        self.tracking = True

                        # Add the detected object to the tracker
                        self.tracker = Camera.tracker_dict['csrt']()
                        self.tracker.init(drone_frame_scaled, tuple(rectangle_scaled))
                        print("Camera clss, detection method: Recognize label: ", label)
                        break

                    return drone_frame
                else:
                    # Contour is a sorted list so all the rest items irrelevant
                    break

        return drone_frame

    # Method manage the prediction
    def prediction_manager(self, drone_frame, rectangle_scaled):

        # Define the scale variable
        scale = Camera.SCALE

        # Get an approximate rectangle coordinates
        (x_min, y_min, box_width, box_height) = [int(a) for a in rectangle_scaled]

        # bounding_boxes contain x1, y1, x2, y2, coordinates and not width and height
        (x_min, y_min, box_width, box_height) = x_min * scale, y_min * scale, box_width * scale, box_height * scale

        rectangle = np.array([x_min, y_min, box_width, box_height])

        # Cutting detected fragment from BGR frame
        cut_fragment_rgb_frame = drone_frame[y_min: y_min + box_height, x_min: x_min + box_width]

        # Resizing frame to the right shape of the model's input
        fragment = cv2.resize(cut_fragment_rgb_frame, (Camera.model_width, Camera.model_height), interpolation=cv2.INTER_CUBIC)

        # Extending dimension from (height, width, channels) to (1, height, width, channels)
        fragment = fragment[np.newaxis, :, :, :]

        # Predict score from model
        scores = self.MODEL.predict(fragment)

        # Finds the labels array index by the max score index of model prediction
        index = np.argmax(scores)

        # Define the label for the cut_fragment from labels array
        label = self.LABELS[index]

        return label

    # Function manage the tracking and return the status
    def tracking_manager(self, drone_frame, drone_frame_scaled, drone_mask_scaled):

        # Set the default status
        self.tracking = False

        # Get the bounding box from the frame
        (success, contour_box) = self.tracker.update(drone_frame_scaled)

        # Keep tracking after the object
        if success:

            # Define the frame boundaries
            width, height = self.width, self.height

            # Get the coordinates of the rectangle around the object
            (x, y, box_width, box_height) = [int(a) for a in contour_box]

            # Check if coordinates is in the frame boundaries
            if 0 <= x and x + box_width <= width and 0 <= y and y + box_height <= height:

                # Define frame scale
                scale = Camera.SCALE

                # Set tracking status ON
                self.tracking = True

                frame_mask = cv2.resize(drone_mask_scaled, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

                # Cut the fragment from the mask frame
                cut_fragment_mask = frame_mask[int(y*scale):int((y+box_height)*scale), int(x*scale):int((x+box_width)*scale)]

                # Cut the fragment from the mask frame
                # cut_fragment_track = tracking_mask[int(y*scale):int((y+h)*scale), int(x*scale):int((x+w)*scale)]

                # Checking if tracking is still running after the object or not
                if self.counter_frames_reading % Camera.tracking_check == 0:

                    # Calculate the pixels values sum, zero means background
                    if np.sum(cut_fragment_mask) < 100 * 255:
                        # Set tracking status OFF
                        self.tracking = False
                        # print("Tracking class: quiet zone")

                # Drawing bounding box on the current BGR frame
                Camera.drawing_target(drone_frame, contour_box)
                # Camera.drawing_rectangle(drone_frame, contour_box)

        return drone_frame  # return contour_box

    # Drawing a target as bounding box around the object
    @staticmethod
    def drawing_target(drone_frame, contour_box):

        # Define frame scale
        scale = Camera.SCALE

        # Get the coordinates of the rectangle around the object
        (x, y, w, h) = [int(a) for a in contour_box]

        # bounding_boxes contain x1, y1, x2, y2, coordinates and not width and height
        x_min, y_min, box_width, box_height = x * scale, y * scale, w * scale, h * scale

        # Define the center point in the bounding box
        x_center, y_center = x_min + (box_width / 2), y_min + (box_height / 2)
        center = (int(x_center), int(y_center))

        cv2.circle(drone_frame, center, 50, (0, 0, 150), 3)
        cv2.line(drone_frame, (center[0] - 75, center[1] - 5), (center[0] - 25, center[1] - 5), (0, 0, 150), 2)
        cv2.line(drone_frame, (center[0] - 75, center[1] + 5), (center[0] - 25, center[1] + 5), (0, 0, 150), 2)
        cv2.line(drone_frame, (center[0] + 25, center[1] - 5), (center[0] + 75, center[1] - 5), (0, 0, 150), 2)
        cv2.line(drone_frame, (center[0] + 25, center[1] + 5), (center[0] + 75, center[1] + 5), (0, 0, 150), 2)

        cv2.line(drone_frame, (center[0] - 5, center[1] - 25), (center[0] - 5, center[1] - 75), (0, 0, 150), 2)
        cv2.line(drone_frame, (center[0] + 5, center[1] - 25), (center[0] + 5, center[1] - 75), (0, 0, 150), 2)
        cv2.line(drone_frame, (center[0] - 5, center[1] + 25), (center[0] - 5, center[1] + 75), (0, 0, 150), 2)
        cv2.line(drone_frame, (center[0] + 5, center[1] + 25), (center[0] + 5, center[1] + 75), (0, 0, 150), 2)

        cv2.putText(drone_frame, "ENEMY", (center[0] - 40, center[1] - 100), cv2.FONT_HERSHEY_PLAIN, 1.7,
                    (0, 0, 255), 1)

    # Drawing a rectangle as bounding box around the object
    @staticmethod
    def drawing_rectangle(drone_frame, contour_box):

        # Define frame scale
        scale = Camera.SCALE

        # Get the coordinates of the rectangle around the object
        (x, y, w, h) = [int(a) for a in contour_box]

        # bounding_boxes contain x1, y1, x2, y2, coordinates and not width and height
        x_min, y_min, box_width, box_height = x * scale, y * scale, w * scale, h * scale

        # Create rectangle object from the boundaries coordinates
        rectangle = np.array([x_min, y_min, box_width, box_height])

        # Drawing bounding box on the current BGR frame
        cv2.rectangle(drone_frame, (x_min, y_min), (x_min + box_width, y_min + box_height), (100, 255, 0), 2)

        # Putting text with label on the current BGR frame
        # cv2.putText(frame, self.detection.label, (x_min - 5, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Design the drone camera frame that plot on the window
    def design_frame(self, frame):

        self.status = 1
        # frame = np.zeros((400, 400, 3), np.uint8)

        # Extract the frame from drone camera
        drone_cam = frame

        if self.camera:
            battery = self.drone.get_battery()
        else:
            battery = 80

        # Safe land when battery is low as 5%
        if battery == 5:
            # Reduce drone speed to 0 before is landing
            print("**WARNING** Battery: ", battery, "%", " landing")

            # Plotting the drone battery on the screen
            cv2.putText(drone_cam, f'{battery}%', (5, self.height - 10), cv2.FONT_HERSHEY_PLAIN, 1.7, (0, 0, 255), 2)
        else:
            # Plotting the drone battery on the screen
            cv2.putText(drone_cam, f'{battery}%', (5, self.height - 10), cv2.FONT_HERSHEY_PLAIN, 1.7, (150, 255, 0), 2)

        # Store the drone movement speed
        drone_speed = str(int(self.drone.motion.movement_speed)) + "cm/s"

        # Plotting the drone battery on the screen
        cv2.putText(drone_cam, f'{drone_speed}', (self.width - 110, self.height - 10), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)

        # Store the drone movement degree
        drone_angle = str(self.drone.drone_angle) + "d/s"

        # Plotting the drone angle on the screen
        cv2.putText(drone_cam, f'{drone_angle}', (self.width - 100, self.height - 40), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)

        # Plotting the drone angle on the screen
        cv2.putText(drone_cam, "Drone Camera", (5, 25), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)

        # Set target around the object on and off
        # if self.target == 1:
        #     center = (200, 200)
        #     cv2.circle(drone_cam, center, 50, (0, 0, 150), 3)
        #     cv2.line(drone_cam, (center[0]-75, center[1]-5), (center[0]-25, center[1]-5), (0, 0, 150), 2)
        #     cv2.line(drone_cam, (center[0]-75, center[1]+5), (center[0]-25, center[1]+5), (0, 0, 150), 2)
        #     cv2.line(drone_cam, (center[0]+25, center[1]-5), (center[0]+75, center[1]-5), (0, 0, 150), 2)
        #     cv2.line(drone_cam, (center[0]+25, center[1]+5), (center[0]+75, center[1]+5), (0, 0, 150), 2)
        #
        #     cv2.line(drone_cam, (center[0]-5, center[1]-25), (center[0]-5, center[1]-75), (0, 0, 150), 2)
        #     cv2.line(drone_cam, (center[0]+5, center[1]-25), (center[0]+5, center[1]-75), (0, 0, 150), 2)
        #     cv2.line(drone_cam, (center[0]-5, center[1]+25), (center[0]-5, center[1]+75), (0, 0, 150), 2)
        #     cv2.line(drone_cam, (center[0]+5, center[1]+25), (center[0]+5, center[1]+75), (0, 0, 150), 2)
        #
        #     cv2.putText(drone_cam, "ENEMY", (center[0]-40, center[1]-100), cv2.FONT_HERSHEY_PLAIN, 1.7, (0, 0, 255), 1)

        return drone_cam, battery

    # Function return 3-Dimension frame
    @staticmethod
    def expands_dimensions(frame):

        new_image = np.zeros((Camera.HEIGHT, Camera.WIDTH, 3), np.uint8)
        new_image[:, :, 0] = frame
        new_image[:, :, 1] = frame
        new_image[:, :, 2] = frame

        return new_image
