
import cv2
import numpy as np

from time import strftime
from timeit import default_timer as timer


# Class process the all the frames
class Process:

    SCALE = 4

    # Define frame size
    WIDTH, HEIGHT = 500, 350

    # Define the thresh hold of the masks
    DIFF_THRESH_HOLD = 20   # Should be low
    MASK_THRESH_HOLD = 100  # Should be high

    # Variables for start rows and cols to put text
    FIRST_COL = 5
    LAST_COL = int(WIDTH - (WIDTH/5))

    FIRST_ROW = 30
    LAST_ROW = int(HEIGHT - 10)

    ROWS_SPACE = 20

    # Define the font size as percent from the screen size
    FONT_SIZE = 0.50

    # Create background of the main frame
    foregroundModel = cv2.createBackgroundSubtractorMOG2()

    def __init__(self, camera):

        # Variable hold all the detections events
        self.log = []
        self.label = ""

        # Define the timer for FPS
        self.fps_start = timer()

        # Variable counting how many times we read frames from camera
        self.counter_frames_reading = 1
        self.counter_frames_tracking = 0
        self.counter_images_processing = 0
        self.counter_frames_processing = 0

        # Define the background
        self.frame = Process.initialize_frame(1)
        self.last_frame = Process.initialize_frame(Process.SCALE)
        self.frame_mask = Process.initialize_frame(1)
        self.inf_frame = Process.initialize_frame(1)
        self.tracking_frame = Process.initialize_frame(1)
        self.drone_camera = Process.initialize_frame(1)
        self.scores_frame = Process.initialize_frame(1)
        self.board_frame = Process.initialize_frame(1)

        if camera is None:
            self.camera_fps = 0
        else:
            # Store the camera fps
            self.camera_fps = int(camera.get(cv2.CAP_PROP_FPS))

        self.program_fps = 0

    # Return black screen
    @staticmethod
    def initialize_frame(scale):
        return np.zeros((int(Process.HEIGHT / scale), int(Process.WIDTH / scale), 3), np.uint8)

    # Method management the image process
    def process_frames(self, camera):

        # Capturing frames one-by-one from camera
        ret, frame = camera.read()

        # If the frame was not retrieved then we break the loop
        if not ret or frame is None:
            system_on = False
            return frame, system_on, 0
        else:
            system_on = True

        # Function return array: [frame, frame_scaled, tracking_mask, frame_mask, last_frame, drone_camera, drone_board]
        frames = Process.preprocess_frames(self, frame, self.last_frame)

        # Extract tracking_mask for the next line
        tracking_mask = frames[2]

        # Start Processing state only if is not quite frames
        pixels_sum = np.sum(tracking_mask)

        return frames, system_on, pixels_sum

    # Function create 3 frames from the frame we read
    def preprocess_frames(self, frame, last_frame):

        # Increase counter every time we read frame from camera
        self.counter_frames_reading += 1

        # Resize the main frame to (WIDTH, HEIGHT) shape
        self.frame = cv2.resize(frame, (Process.WIDTH, Process.HEIGHT))

        # Copy frame to work with different variable
        self.board_frame = cv2.resize(self.board_frame, (Process.WIDTH, Process.HEIGHT))

        # Copy frame to work with different variable
        frame_rgb = cv2.resize(frame, (int(Process.WIDTH / Process.SCALE), int(Process.HEIGHT / Process.SCALE)))

        # Return mask frame
        self.frame_mask = Process.cv_mask(frame_rgb, last_frame)
        # self.frame_mask = Process.mask(frame_rgb, last_frame)
        # self.frame_mask = Process.mask_tracking(frame_rgb, last_frame)

        # Create tracking mask which more sensitive to noises
        tracking_mask = Process.mask_tracking(frame_rgb, last_frame)

        # Copy frame to work with different variable
        self.drone_camera = cv2.resize(self.drone_camera, (Process.WIDTH, Process.HEIGHT))

        # Define last frame
        self.last_frame = frame_rgb.copy()

        # Collect all the frames into 1 array
        frames = [self.frame, frame_rgb, tracking_mask, self.frame_mask, self.last_frame, self.drone_camera, self.board_frame]

        return frames

    # Function define the mask
    @staticmethod
    def mask(frame_rgb, background):

        # Converting captured frame to GRAY by OpenCV function
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # Create one more frame with Gaussian blur
        frame_gray = cv2.GaussianBlur(frame_gray, (25, 25), 0)

        # Converting captured frame to GRAY by OpenCV function
        background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

        # Create one more frame with Gaussian blur
        background = cv2.GaussianBlur(background, (25, 25), 0)

        # Return mask to detect change between two frames
        abs_diff = cv2.absdiff(frame_gray, background)

        # Function exclude values that ara more than threshold = 15 0 and more than 255
        _, mask = cv2.threshold(abs_diff, 20, 255, cv2.THRESH_BINARY)

        # Dilates the object in the frame
        dilated_mask = cv2.dilate(mask, None, iterations=5)

        return dilated_mask

    # Function create a mask to track the object
    @staticmethod
    def mask_tracking(frame_rgb, last_frame):

        # Converting captured frame to GRAY by OpenCV function
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # Converting captured frame to GRAY by OpenCV function
        last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

        # Return mask to detect change between two frames
        abs_diff = cv2.absdiff(frame_gray, last_frame)

        # Function exclude values that ara more than threshold = 15 0 and more than 255
        _, abs_diff_mask = cv2.threshold(abs_diff, Process.DIFF_THRESH_HOLD, 255, cv2.THRESH_BINARY)

        # Expend mask dimension to 3 dimension
        # mask_frame = Process.dimensional_expansion(abs_diff_mask)
        mask_frame = abs_diff_mask

        return mask_frame

    # Function create a mask with connectedComponents
    @staticmethod
    def cv_mask(frame_rgb, last_frame):

        # Apply the frame to foreground model
        foreground_mask = Process.foregroundModel.apply(frame_rgb)

        # Reduce noises
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        foreground_mask = cv2.morphologyEx(np.float32(foreground_mask), cv2.MORPH_OPEN, structuring_element)

        # Find out connected components and keep only the large components
        num_labels, image_labels = cv2.connectedComponents(np.array(0 < foreground_mask, np.uint8))

        # Return components larger than threshold
        foreground_mask = Process.keep_large_components(image_labels, threshold=0)

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

    # Function return 3-Dimension frame
    @staticmethod
    def dimensional_expansion(frame):

        new_image = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
        new_image[:, :, 0] = frame
        new_image[:, :, 1] = frame
        new_image[:, :, 2] = frame

        return new_image

    # Function get 3 frames and collect them to 1 frame
    @staticmethod
    def frames_collection(left_frame, mid_frame, right_frame):

        # Insert all frames to array for scan it
        frames = [left_frame, mid_frame, right_frame]

        # Change all frames to 3 channels
        for i in range(len(frames)):

            # Check if frames[i][3] is exit
            if len(frames[i].shape) < 3:
                # Adding 3-Dimension to the image
                frames[i] = frames[i][:, :, np.newaxis]

            # Find frames that not 3 channels
            if frames[i].shape[2] != 3:
                # Function expand mask's dimension from 1 to 3 dimensions
                frames[i] = np.repeat(frames[i], 3, axis=2)

        # Define frames in the right order
        left_frame = frames[0]
        mid_frame = frames[1]
        right_frame = frames[2]

        # Create one window that contain: frame, tracking, mask
        collection_frame = np.hstack((left_frame, mid_frame, right_frame))

        return collection_frame

    # Function create the information frame in the main window
    def build_info_frame(self, detection_obj):

        time = detection_obj.detection_time

        # Store camera fps
        fps = self.camera_fps

        counter_frames_reading = self.counter_frames_reading
        counter_frames_tracking = self.counter_frames_tracking
        counter_images_processing = self.counter_images_processing

        # Define the frame boundaries
        left_boundary = Process.FIRST_COL
        row_space = Process.ROWS_SPACE

        # Variable for ain boundaries
        shift_left = int(Process.WIDTH / 4) + 10
        shift_down = int(Process.HEIGHT - Process.FIRST_ROW * 4)

        # Define the font size as 2*FONT_SIZE=(HEIGHT/1000)
        font_size = (Process.FONT_SIZE * 1)

        # Initialize frame with white background
        inf_frame = Process.empty_frame(self.inf_frame)

        # Variable hold amount images processing in percentage
        processing_percentage = str(int((counter_images_processing * 100) / counter_frames_reading)) + "%"
        tracking_percentage = str(int((counter_frames_tracking * 100) / counter_frames_reading)) + "%"

        # Adding text with right time for current frame
        cv2.putText(inf_frame, strftime("%H:%M:%S"), (Process.LAST_COL, Process.LAST_ROW - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1, cv2.LINE_AA)

        # Adding text with right date for current frame
        cv2.putText(inf_frame, strftime("%d/%m/%Y"), (Process.LAST_COL - 30, Process.LAST_ROW),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1, cv2.LINE_AA)

        # Adding text with Model name of the app
        cv2.putText(inf_frame, "Model: ResNet50", (Process.FIRST_COL, Process.FIRST_ROW - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1, cv2.LINE_AA)

        # Adding text with the training dataset of the app
        cv2.putText(inf_frame, "Dataset: Cifar10", (Process.FIRST_COL, Process.FIRST_ROW - 10 + Process.ROWS_SPACE),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1, cv2.LINE_AA)

        # Adding text with Camera FPS for current frame
        cv2.putText(inf_frame, 'Camera FPS: ' + '{0:.0f}'.format(fps),
                    (Process.FIRST_COL, Process.FIRST_ROW +  - 10 + Process.ROWS_SPACE * 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1, cv2.LINE_AA)

        # Adding text with reading frames counter for current frame
        cv2.putText(inf_frame, 'Reading Frames: ' + '{0:.0f}'.format(counter_frames_reading),
                    (Process.FIRST_COL, Process.FIRST_ROW - 10 + Process.ROWS_SPACE * 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1, cv2.LINE_AA)

        # Adding text with time spent for 2D convolution for current frame
        cv2.putText(inf_frame, 'Tracking Images: ' + tracking_percentage,
                    (Process.FIRST_COL, Process.FIRST_ROW - 10 + Process.ROWS_SPACE * 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1, cv2.LINE_AA)

        # Adding text with processing images counter for current frame
        cv2.putText(inf_frame, 'Processing Images: ' + processing_percentage,
                    (Process.FIRST_COL, Process.FIRST_ROW - 10 + Process.ROWS_SPACE * 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1, cv2.LINE_AA)

        # Define the next line where the detections text will start
        end_text_line = (Process.FIRST_ROW + (Process.ROWS_SPACE * 6))

        # Function put text of all detections events on the frame
        Process.plotting_detections(self, inf_frame, self.label, time, end_text_line, detection_obj)

        self.inf_frame = inf_frame

    # Function takes 6 windows and collect them to one main window
    def build_main_window(self, detection_obj, drone_obj):

        # Function return 3 frames collect to 1 window
        upper_window = Process.build_upper_window(self, detection_obj)

        # Function return 3 frames collect to 1 window
        lower_window = Process.build_lower_window(self, drone_obj)

        # Create one window that contain: upper_window and lower_window
        main_window = np.vstack((upper_window, lower_window))

        return main_window

    # Method build the upper window
    def build_upper_window(self, detection_obj):

        # Function build frame that contain the app stats and information
        Process.build_info_frame(self, detection_obj)

        # Function return one window that contain (frame_bgr, track_frame, mask_frame)
        upper_window = Process.frames_collection(self.frame, self.inf_frame, self.tracking_frame)

        return upper_window

    # Method build the lower window
    def build_lower_window(self, drone_obj):

        # Function display frames on the screen
        self.drone_camera, self.board_frame, drone_battery = Process.build_drone_frames(drone_obj)

        # Function return one window that contain (cut_fragment_bgr_frame, scores_frame, mask_frame)
        lower_window = Process.frames_collection(self.drone_camera, self.scores_frame, self.board_frame)

        return lower_window

    # Function management the main window
    @staticmethod
    def build_drone_frames(drone_obj):

        drone_frame, drone_mask = drone_obj.read_frame_from_camera()

        # Adding some features to frame window
        drone_cam, battery = drone_obj.drone_camera.design_frame(drone_frame)

        # Variable store the board that drawing the drone movements
        drone_board = drone_obj.board.board

        # Cleaning some pixels for the next iterate
        if drone_obj.on_tracking:
            # point = (drone_obj.board.x_roi, drone_obj.board.y_roi)
            # drone_obj.board.board[point[1] + 10:point[1] + 40, point[0] + 10:point[0] + 200] = 0
            # print(point)
            pass

        return drone_cam, drone_board, battery

    def display_window(self, counter_fps, detection_obj, drone_obj):

        "frame_mask shape is scaled"
        self.frame_mask = cv2.resize(self.frame_mask, (Process.WIDTH, Process.HEIGHT), interpolation=cv2.INTER_CUBIC)

        # Function manage the frames reader variables and get input for user
        counter_fps, key, system_on = Process.reader_manger(self, counter_fps, detection_obj, drone_obj)

        # Create one window that contain: upper_window and lower_window
        main_window = Process.build_main_window(self, detection_obj, drone_obj)

        # Plotting all the frames in one window
        cv2.imshow("Main_Window", main_window)

        return counter_fps, key, system_on

    # Function manage the frames reader variables like fps etc.
    def reader_manger(self, counter_fps, detection_obj, drone_obj):

        # Stopping the timer for FPS
        fps_stop = timer()

        # Define FPS
        # program_fps = counter_fps

        # Print FPS every 1 second
        if 1.0 <= fps_stop - self.fps_start:

            # Define FPS
            # program_fps = counter_fps

            # Store the program fps
            self.program_fps = counter_fps

            # Restart timer for FPS
            self.fps_start = timer()

            # Reset FPS counter
            counter_fps = 0

        # Function waits for key to be pressed
        key = cv2.waitKeyEx(1)  # % 256

        system_on = drone_obj.remote_control(key, detection_obj)

        return counter_fps, key, system_on

    # Function continue info_frame Function and put text of detection in info frame
    def plotting_detections(self, inf_frame, label, time, start_text_line, detection_obj):

        label = detection_obj.label

        # Define the font size as 2*FONT_SIZE=(HEIGHT/1000)
        font_size = Process.FONT_SIZE

        # Find max label string length
        max_len = len('airplane')

        # Checking if there is detection
        if 0 < len(label):

            # Order all line to the same length
            if len(label) < max_len:
                label = str(label) + (max_len - len(label)) * " "

            detection = "Detected " + str(label) + " at: " + str(time)

            # Insert first element to log array
            if len(self.log) == 0:
                self.log.append(detection)
            else:
                # Check there ara different time of detection event
                if detection != self.log[-1]:
                    # Calculate how many minutes passed from the last detection
                    time_past_in_minutes = Process.detection_timer(time, self.log[-1])
                    self.log.append(detection)

        # Line number we start write objects we detected
        line = start_text_line

        # Define the end line boundaries
        end_line = (Process.LAST_ROW - 20)

        # Scan all the detections object to plot them on the frame
        for i in range(len(self.log)):

            # Variable represent the detection event as a string
            event = self.log[i]

            # Check frames boundaries including text
            if end_line <= line:
                # Delete old detection by set pixels to while and initialize line to start
                inf_frame[start_text_line - 20:end_line+5, :] = 255

                # Back to line 130
                line = start_text_line

            # Adding text with DETECTION EVENT for current frame
            cv2.putText(inf_frame, event, (2, line), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 1, cv2.LINE_AA)

            # skip to the next line
            line += int(Process.ROWS_SPACE)

    @staticmethod
    def detection_timer(detection_time, last_time):

        if 0 < len(last_time):
            element = last_time
            minute = element[len(element) - 2]
            minute = int(minute) * 10
            second = element[len(element) - 1]
            second = int(second)
            minutes = minute + second
            last_time_minutes = minutes

            element = detection_time
            minute = element[len(element) - 2]
            minute = int(minute) * 10
            second = element[len(element) - 1]
            second = int(second)
            minutes = minute + second
            detection_time_minutes = minutes

            time_past = abs(detection_time_minutes - last_time_minutes)

            return time_past
        else:
            return 0

    # Function return empty frame for initialize the main window
    @staticmethod
    def empty_frame(frame_bgr):

        empy_frame = np.zeros(frame_bgr.shape, np.uint8)
        empy_frame[:, :, 0] = 255
        empy_frame[:, :, 1] = 255
        empy_frame[:, :, 2] = 255

        return empy_frame

