import io
import os
import cv2

import winsound
import numpy as np
# from cv2 import cv2
# import cv2.cv2 as cv2
from time import strftime
from collections import deque
from timeit import default_timer as timer

import matplotlib.pyplot as plt

from Process import Process


# Class detection object from webcam
class Detection:

    # Define objects boundaries size
    MIN_OBJECT_AREA = 50
    MAX_OBJECT_AREA = 1000  # (Process.WIDTH * Process.HEIGHT * 25) / 100

    # Define variables for height and width shape to prediction model input
    model_width, model_height = 224, 224

    # Define tracker dictionary
    tracker_dict = {'csrt': cv2.legacy.TrackerCSRT_create,
                    'kcf': cv2.legacy.TrackerKCF_create,
                    'mil': cv2.legacy.TrackerMIL_create,
                    'tld': cv2.legacy.TrackerTLD_create(),
                    'medianflow': cv2.legacy.TrackerMedianFlow_create(),
                    'boosting': cv2.legacy.TrackerMOSSE_create(),
                    'mosse': cv2.legacy.TrackerMOSSE_create()}

    # This path is location for the sound file
    soundPath = r'Sounds/Falcon/Falcon.mp3'

    home_folder = r'C:\WorkSpace\JupyterNotebook\Final Projects\Expel Pigeons'

    # This path is location for the saved images
    outPutPath = r'C:\WorkSpace\JupyterNotebook\Final Projects\Expel Pigeons\Files\Saved Images\Detection Events'

    # This path is location for the saved detections events
    txtPath = r'C:\WorkSpace\JupyterNotebook\Final Projects\Expel Pigeons\Files\Saved Images\Detection Events\Detection Events.txt'

    def __init__(self, process, model, model_labels):

        # Initialize variables
        self.label = ""
        self.status = "Detecting"
        self.detection_time = " "

        # Process class managements frames
        self.process = process

        # Variables send from Process class
        self.scale = Process.SCALE
        self.width = Process.WIDTH
        self.height = Process.HEIGHT

        # Define the tracking status
        self.tracking_on = False

        # Initializing deque object for center points of the detected object
        self.points = deque(maxlen=50)

        # Define the prediction model and it labels of the object
        self.model = model
        self.model_labels = model_labels

        # Write title on the file
        Detection.write_detection_event()

        # Counting how many birds detected
        self.counter_birds_predictions = 0

        # Counting how many frames predicted
        self.counter_frames_predictions = 0

        # Counting how many failed predicted
        self.counter_failed_predictions = 0

        # Initialize our tracker after the object
        self.tracker = Detection.tracker_dict['csrt']()

        # self.score = 0
        # self.prediction_timer = 0

    # Function design the detections events file
    @staticmethod
    def write_detection_event():

        # Writing The Headline of the text file
        with open(Detection.txtPath, 'a') as f:
            f.write("###################### Detection Events #########################")
            f.write('\n')
            f.write('\n')

    # Function manage the detection and return status and coordinates
    def detection_manager(self, frame, frame_scaled, frame_mask):

        # Initialize label variables
        self.status = "Detecting"

        # Function return array of all contours we found
        contours, _ = cv2.findContours(frame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sorted the contours and define the larger first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Scan the contours list
        for contour in contours:

            # Return square area of the given contour
            contour_area = cv2.contourArea(contour)

            # Find contours between MIN_OBJECT_AREA to MAX_OBJECT_AREA
            if contour_area < Detection.MAX_OBJECT_AREA:
                if Detection.MIN_OBJECT_AREA < contour_area:

                    # Increase prediction counter
                    self.counter_frames_predictions += 1

                    # Function return the cutted rgb fragment in without resizing
                    cut_fragment_rgb, rectangle, rectangle_scaled = Detection.cut_fragment(self, frame, contour)

                    # Function predict the cutted fragment and return the result
                    scores, label = Detection.prediction_manager(self, cut_fragment_rgb)

                    # Start tracking after the detected bird
                    if label == "bird":

                        # Store the label we detected
                        self.label = label

                        # Store the score of the detection
                        self.score = scores[0][np.argmax(scores)]

                        # Increasing Birds prediction counter
                        self.counter_birds_predictions += 1

                        # Function add object to the tracker, save detection and drawing rectangle
                        self.process.frame, frame_scaled, self.process.scores_frame, self.tracker = Detection. \
                            detected_bird(self, frame, rectangle, frame_scaled, rectangle_scaled, scores)

                        # Update tracking status
                        self.tracking_on = True
                        break

                    # End of if not detected birds
                    else:
                        # Initialize the label we detected
                        self.label = label

                        # Increasing fail predictions counter
                        self.counter_failed_predictions = self.counter_frames_predictions-self.counter_birds_predictions
                else:
                    # Contour is a sorted list so all the rest items irrelevant
                    break

        return self.process.frame, self.tracking_on

    # Function cut the detected fragment and return it
    def cut_fragment(self, frame, contour):

        # Define the scale variable
        scale = self.scale

        # Get an approximate rectangle coordinates
        (x_min, y_min, box_width, box_height) = cv2.boundingRect(contour)

        # Create rectangle object from the boundaries coordinates of scaled frames
        rectangle_scaled = np.array([x_min, y_min, box_width, box_height])

        # bounding_boxes contain x1, y1, x2, y2, coordinates and not width and height
        (x_min, y_min, box_width, box_height) = x_min * scale, y_min * scale, box_width * scale, box_height * scale

        # Cutting detected fragment from BGR frame
        cut_fragment_rgb_frame = frame[y_min: y_min + box_height, x_min: x_min + box_width]

        # Create rectangle object from the boundaries coordinates
        rectangle = np.array([x_min, y_min, box_width, box_height])

        return cut_fragment_rgb_frame, rectangle, rectangle_scaled

    # Function manage the prediction part and return prediction results
    def prediction_manager(self, cut_fragment_rgb):

        # Measuring classification time
        start = timer()

        # Function return all scores of model predictions
        scores = Detection.prediction_model(self, cut_fragment_rgb)

        # End of Measuring classification time
        end = timer()

        # Current time of detection object
        self.detection_time = strftime("%d/%m/%Y %H:%M:%S")

        # Calculate the time that needed to predict the fragment
        self.prediction_timer = end - start

        # Finds the labels array index by the max score index of model prediction
        index = np.argmax(scores)

        # Define the label for the cut_fragment from labels array
        label = self.model_labels[index]

        return scores, label

    # Function predict model's output from the cutted fragment
    def prediction_model(self, cut_fragment_rgb):

        # Create a copy of the cut_fragment_bgr frame
        fragment = cut_fragment_rgb.copy()

        # Resizing frame to the right shape of the model's input
        fragment = cv2.resize(fragment, (Detection.model_width, Detection.model_height), interpolation=cv2.INTER_CUBIC)

        # Extending dimension from (height, width, channels) to (1, height, width, channels)
        fragment = fragment[np.newaxis, :, :, :]

        # Predict score from model
        scores = self.model.predict(fragment)

        return scores

    # Function drawing rectangle around the predicted object
    @staticmethod
    def drawing_rectangle(frame, rectangle, label):

        # Get an approximate rectangle coordinates
        (x_min, y_min, box_width, box_height) = [int(a) for a in rectangle]

        # Drawing bounding box on the current BGR frame
        cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), (0, 255, 0), 3)

        # Putting text with label on the current BGR frame
        cv2.putText(frame, label, (x_min - 5, y_min - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        return frame

        # Function initialize variables of bird detected

    def detected_bird(self, frame, rectangle, frame_scaled, rectangle_scaled, scores):

        # Initialize Deque points
        self.points.clear()

        # Makes a sound to alert about bird
        # _thread.start_new_thread(winsound.PlaySound(soundPath, winsound.SND_FILENAME))

        # Create scaled frame with rectangle scaled for the Tracker object
        frame_scaled = Detection.drawing_rectangle(frame_scaled, rectangle_scaled, "")

        # Function return a bgr frame with rectangle around the cut fragment
        frame = Detection.drawing_rectangle(frame, rectangle, self.label)

        # Save detection event in date-time format
        Detection.save_detection_event(self, frame)

        # Function return a frame with all the labels scores
        scores_frame = Detection.bar_chart(self, scores[0], self.model_labels)

        # Add the detected object to the tracker
        self.tracker = Detection.tracker_dict['csrt']()
        self.tracker.init(frame_scaled, tuple(rectangle_scaled))

        return frame, frame_scaled, scores_frame, self.tracker

    # Function save the frame of the detected event
    def save_detection_event(self, frame):

        detection = ""
        label = self.label
        detection_time = self.detection_time

        # Create a copy of time object
        time = str(detection_time)

        # Save detection_time in format that fit to files
        time = time[:2] + "-" + time[3:5] + "-" + time[6:13] + "-" + time[14:16] + "-" + time[17:]

        # Define the name of the image
        image_name = label + ' ' + str(time) + '.jpg'

        # Define the file address
        final_path = os.path.join(Detection.outPutPath, image_name)

        # Save the frame of detection event
        cv2.imwrite(final_path, frame)

        # Checking if there is detection
        if 0 < len(label):

            # Order all line to the same length
            if len(label) < len("airplane"):
                label = str(label) + (len("airplane") - len(label)) * " "

            # Create string of detection event in date-time format
            detection = "Detected " + str(label) + " at: " + detection_time

        # Writing the detection event into the file in txtPath location
        with open(Detection.txtPath, 'a') as f:
            f.write(detection)

    # Function plot bar chart with scores values
    def bar_chart(self, obtained_scores, classes_names):

        # Arranging X axis
        x_positions = np.arange(obtained_scores.size)

        # Creating bar chart
        bars = plt.bar(x_positions, obtained_scores, align='center', alpha=0.6)

        # Highlighting the highest bar
        bars[np.argmax(obtained_scores)].set_color('red')

        # Giving labels to bars along X axis
        plt.xticks(x_positions, classes_names, rotation=25, fontsize=10)

        # Giving names to axes
        plt.xlabel('Class', fontsize=20)
        plt.ylabel('Value', fontsize=20)

        # Giving name to bar chart
        plt.title('Obtained Scores', fontsize=20)

        # Adjusting borders of the plot
        plt.tight_layout(pad=2.5)

        # Initializing object of the buffer
        b = io.BytesIO()

        # Saving bar chart into the buffer
        plt.savefig(b, format='png', dpi=200)

        # Closing plot with bar chart
        plt.close()

        # Moving pointer to the beginning of the buffer
        b.seek(0)

        # Reading bar chart from the buffer
        bar_image = np.frombuffer(b.getvalue(), dtype=np.uint8)

        # Closing buffer
        b.close()

        # Decoding buffer
        bar_image = cv2.imdecode(bar_image, 1)

        # Resize frame to HEIGHT X WIDTH
        bar_image = cv2.resize(bar_image, (self.width, self.height))

        # Returning Numpy array with bar chart
        return bar_image

    # Function writes all the app's information into a text file
    def write_info_txt(self):

        counter_frames_reading = self.process.counter_frames_reading
        counter_frames_tracking = self.process.counter_frames_tracking
        counter_frames_processing = self.process.counter_frames_processing
        counter_images_processing = self.process.counter_images_processing
        counter_birds_predictions = self.counter_birds_predictions
        counter_frames_predictions = self.counter_frames_predictions

        counter_fail_predictions = counter_frames_predictions - counter_birds_predictions
        counter_frames_not_processing = counter_frames_processing - counter_images_processing

        # Stores all the application's information in array and then writes them into a file
        app_info = []
        app_info.append(str("\n###################### Application Information ##################\n"))
        app_info.append(str("counter frames reading: " + str(counter_frames_reading)))
        app_info.append(str("counter frames tracking: " + str(counter_frames_tracking)))
        app_info.append(str("counter frames processing: " + str(counter_frames_processing)))
        app_info.append(str("counter images processing: " + str(counter_images_processing)))
        app_info.append(str("counter frames predictions: " + str(counter_frames_predictions)))
        app_info.append(str("counter birds predictions: " + str(counter_birds_predictions)))
        app_info.append(str("counter fail predictions: " + str(counter_fail_predictions)))
        app_info.append(str("counter frames not processing: " + str(counter_frames_not_processing)))
        app_info.append(str(" "))
        app_info.append(str("Tracking Frames: " + str(int((counter_frames_tracking * 100) /
                                                          counter_frames_reading)) + "%"))
        app_info.append(str("Processing Frames: " + str(int((counter_frames_processing * 100) /
                                                            counter_frames_reading)) + "%"))
        app_info.append(str("Right Predictions: " + '{0:.0f}'.format((counter_birds_predictions * 100) /
                                                                     counter_frames_predictions) + "%"))
        app_info.append(str(" "))
        app_info.append(str("############################## END ################################\n\n"))

        # Writing all application's information into a text file
        with open(Detection.txtPath, 'a') as f:
            f.write('\n'.join(app_info))
