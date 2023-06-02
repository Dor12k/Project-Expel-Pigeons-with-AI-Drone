
import cv2
import numpy as np

from Process import Process


# Class management the tracking after objects part
class Tracking:

    # Define timer to check the tracking
    tracking_check = 10

    def __init__(self, process, detection):

        self.process = process
        self.detection = detection

        self.scale = process.SCALE
        self.width = process.WIDTH
        self.height = process.HEIGHT

    # Function manage the tracking and return the status
    def tracking_manager(self, frame, frame_scaled, frame_mask, tracking_mask):

        # Set the default status
        self.detection.tracking_on = False

        # Get the bounding box from the frame
        (success, contour_box) = self.detection.tracker.update(frame_scaled)

        # Keep tracking after the object
        if success:

            # Define the frame boundaries
            width, height = self.width, self.height

            # Get the coordinates of the rectangle around the object
            (x, y, w, h) = [int(a) for a in contour_box]

            # Check if coordinates is in the frame boundaries
            if 0 <= x and x + w <= width and 0 <= y and y + h <= height:

                # Define frame scale
                scale = self.scale

                # Set tracking status ON
                self.detection.tracking_on = True

                frame_mask = cv2.resize(frame_mask, (Process.WIDTH, Process.HEIGHT), interpolation=cv2.INTER_CUBIC)
                tracking_mask = cv2.resize(tracking_mask, (Process.WIDTH, Process.HEIGHT), interpolation=cv2.INTER_CUBIC)

                # Cut the fragment from the mask frame
                cut_fragment_mask = frame_mask[int(y*scale):int((y+h)*scale), int(x*scale):int((x+w)*scale)]

                # Cut the fragment from the mask frame
                cut_fragment_track = tracking_mask[int(y*scale):int((y+h)*scale), int(x*scale):int((x+w)*scale)]

                # Checking if tracking is still running after the object or not
                if self.process.counter_frames_reading % Tracking.tracking_check == 0:

                    # Calculate the pixels values sum, zero means background
                    if np.sum(cut_fragment_track) < 100*255:
                        # Set tracking status OFF
                        self.detection.tracking_on = False
                        # print("Tracking class: quiet zone")

                # Drawing bounding box on the current BGR frame
                Tracking.drawing_rectangle(self, frame, contour_box)

        # Function return a frame with the tracking of the cut fragment
        self.process.tracking_frame = Tracking.drawing_tracking(self, frame, contour_box)

        return frame, self.process.tracking_frame  # return contour_box

    # Function create frame that follow the object movement
    def drawing_tracking(self, frame, contour_box):

        # Define frame scale
        scale = self.scale

        # Variable for ain boundaries
        shift_left = int(Process.WIDTH / 5) + 10
        shift_down = int(Process.HEIGHT - Process.FIRST_ROW)

        # Define the prediction time for fragment
        model_prediction_time = self.detection.prediction_timer

        # Get the coordinates of the rectangle around the object
        (x, y, w, h) = [int(a) for a in contour_box]

        # bounding_boxes contain x1, y1, x2, y2, coordinates and not width and height
        x_min, y_min, box_width, box_height = x * scale, y * scale, w * scale, h * scale

        # Create rectangle object from the boundaries coordinates
        rectangle = np.array([x_min, y_min, box_width, box_height])

        # Getting current center coordinates of the bounding box
        center = (int(x_min + box_width / 2), int(y_min + box_height / 2))

        # Adding current point to the queue
        self.detection.points.appendleft(center)

        # Creating image with black background
        track_frame = np.zeros(frame.shape, np.uint8)

        # Changing background to Black color
        track_frame[:, :, 0] = 0
        track_frame[:, :, 1] = 0
        track_frame[:, :, 2] = 0

        # Visualizing tracker line
        for i in range(1, len(self.detection.points)):

            # If no points collected yet
            if self.detection.points[i - 1] is None or self.detection.points[i] is None:
                continue

            # Draw the line between points
            cv2.line(track_frame, self.detection.points[i - 1], self.detection.points[i], (50, 200, 50), 2)

        # Adding text with center coordinates of the bounding box
        cv2.putText(track_frame, 'X: {0}'.format(center[0]), (Process.FIRST_COL, Process.FIRST_ROW),
                    cv2.FONT_HERSHEY_SIMPLEX, Process.FONT_SIZE * 1.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(track_frame, 'Y: {0}'.format(center[1]), (Process.FIRST_COL, Process.FIRST_ROW + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, Process.FONT_SIZE * 1.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Adding text with time spent for 2D convolution for current frame
        cv2.putText(track_frame, 'Time : ' + '{0:.3f}'.format(model_prediction_time),
                    (Process.FIRST_COL, Process.LAST_ROW - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    Process.FONT_SIZE, (255, 255, 255), 1, cv2.LINE_AA)

        # Adding text with score of convolution for current frame
        cv2.putText(track_frame, 'Score : ' + '{0:.3f}'.format(self.detection.score),
                    (Process.FIRST_COL, Process.LAST_ROW),
                    cv2.FONT_HERSHEY_SIMPLEX, Process.FONT_SIZE, (255, 255, 255), 1, cv2.LINE_AA)

        # Adding text with current label on the frame
        cv2.putText(track_frame, "FPS: " + str(self.process.program_fps + 1),
                    (Process.LAST_COL, Process.LAST_ROW),
                    cv2.FONT_HERSHEY_SIMPLEX, Process.FONT_SIZE, (255, 255, 255), 1, cv2.LINE_AA)

        # If Tracking is on - put text on frame
        if self.detection.tracking_on:
            # Adding text with tracking status on the frame
            cv2.putText(track_frame, 'Tracking ', (Process.LAST_COL - 10, Process.FIRST_ROW),
                        cv2.FONT_HERSHEY_TRIPLEX, Process.FONT_SIZE, (50, 200, 50), 1, cv2.LINE_AA)

        # Delete the "Tracking" alert from the screen
        if not self.detection.tracking_on:
            track_frame[Process.WIDTH - 135:, 0:Process.FIRST_ROW + 15, 0] = 0
            track_frame[Process.WIDTH - 135:, 0:Process.FIRST_ROW + 15, 1] = 0
            track_frame[Process.WIDTH - 135:, 0:Process.FIRST_ROW + 15, 2] = 0

            track_frame[center[1], center[0], 0] = 0
            track_frame[center[1], center[0], 1] = 0
            track_frame[center[1], center[0], 2] = 0

            self.detection.points.clear()

        return track_frame

    # Function drawing rectangle around the objects
    def drawing_rectangle(self, frame, contour_box):

        # Define frame scale
        scale = self.scale

        # Get the coordinates of the rectangle around the object
        (x, y, w, h) = [int(a) for a in contour_box]

        # bounding_boxes contain x1, y1, x2, y2, coordinates and not width and height
        x_min, y_min, box_width, box_height = x * scale, y * scale, w * scale, h * scale

        # Create rectangle object from the boundaries coordinates
        rectangle = np.array([x_min, y_min, box_width, box_height])

        # Drawing bounding box on the current BGR frame
        cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), (100, 255, 0), 2)

        # Putting text with label on the current BGR frame
        cv2.putText(frame, self.detection.label, (x_min - 5, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

