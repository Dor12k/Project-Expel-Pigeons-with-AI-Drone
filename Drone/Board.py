
import cv2
import numpy as np

from time import sleep
from Process import Process


# Board Class represent the drone movement in the environment
class Board:

    def __init__(self, board_frame):

        width, height = board_frame.shape[1], board_frame.shape[0]

        # Define the board size
        self.width = width
        self.height = height
        self.board = np.zeros((self.height, self.width, 3), np.uint8)

        # Define the start location
        self.start_x = int(self.width/2)
        self.start_y = int(self.height/2)
        self.start_coordinate = [self.start_x, self.start_y]

        self.x_roi = self.start_x
        self.y_roi = self.start_y

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, board_width):
        self.__width = board_width

    @property
    def height(self):
        return self.__height

    @height.setter
    def height(self, board_height):
        self.__height = board_height

    # Return np.zero()
    @property
    def board(self):
        return self.__board

    # Define board
    @board.setter
    def board(self, drone_board):
        self.__board = drone_board

    @property
    def start_x(self):
        return self.__start_x

    @start_x.setter
    def start_x(self, x):
        self.__start_x = x

    @property
    def start_y(self):
        return self.__start_y

    @start_y.setter
    def start_y(self, y):
        self.__start_y = y

    @property
    def start_coordinate(self):
        return self.start_coordinate

    @start_coordinate.setter
    def start_coordinate(self, point):
        self.__start_coordinate = point
        self.__start_x = point[0]
        self.__start_y = point[1]

    # Draw the drone keyboards track on the board
    def draw_points(self, coordinates, drone_angle):

        # Plotting the title frame on the screen
        cv2.putText(self.board, "Drone board",
                    (Process.FIRST_COL, Process.FIRST_ROW), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

        # Initialize ROI pixels of the last text
        self.board[self.y_roi + 10:self.y_roi + 40, self.x_roi + 10:self.x_roi + 200] = 0

        # Fix error
        n = drone_angle
        n += 1

        # Tag patrolling on the screen when drone is patrolling
        text_coordinate = (Process.LAST_COL - 15, Process.FIRST_ROW)
        cv2.putText(self.board, 'Patrolling ', text_coordinate, cv2.FONT_HERSHEY_TRIPLEX,
                    Process.FONT_SIZE, (50, 200, 50), 1, cv2.LINE_AA)

        # Check coordinates array length
        last_point = coordinates[0]

        # Scan all the coordinates and draw it on the board
        for coordinate in coordinates:

            # Every 25 cells is 1 meter. add 0.5 to round up / down
            x_coordinate = round((coordinate[0] - self.start_x))
            y_coordinate = round((coordinate[1] - self.start_y))

            # Define the coordinates as a point
            point = ((self.start_x + x_coordinate), int(self.start_y + y_coordinate))

            # Check if the point coordinates is in the board boundaries
            if (0 < point[0] < self.width) or (0 < point[1] < self.height):
                cv2.line(self.board, last_point, point, (0, 255, 0), 5)

            # Update last point
            last_point = point
            # sleep(0.01)

        # Draw the last point in the track
        x_coordinate = round((coordinates[-1][0] - self.start_x))
        y_coordinate = round((coordinates[-1][1] - self.start_y))

        point = ((self.start_x + x_coordinate), int(self.start_y + y_coordinate))

        # Check boundaries
        if (0 < point[0] < self.width) or (0 < point[1] < self.height):
            # Draw the triangle on the board
            # triangle_cnt = np.array([(point[0], point[1] - 20), (point[0] - 10, point[1]), (point[0] + 10, point[1])])
            # cv2.drawContours(self.board, [triangle_cnt], 0, (255, 255, 255), -1)
            pass

        cv2.putText(self.board,
                    f'({(coordinates[-1][0] - self.start_x) / 25}, {-(coordinates[-1][1] - self.start_y) / 25})m',
                    (int(self.start_x + x_coordinate) + 10, int(self.start_y + y_coordinate) + 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1)

        self.x_roi = point[0]
        self.y_roi = point[1]

    # Draw the drone track on the board
    def draw_tracks(self, coordinates, drone_angle):

        # Plotting the title frame on the screen
        cv2.putText(self.board, "Drone board",
                    (Process.FIRST_COL, Process.FIRST_ROW), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

        # Clean pixels of last time we called the method
        self.board[self.y_roi + 10:self.y_roi + 40, self.x_roi + 10:self.x_roi + 200] = 0

        # Tag patrolling on the screen when drone is patrolling
        text_coordinate = (Process.LAST_COL - 15, Process.FIRST_ROW)
        cv2.putText(self.board, 'Patrolling ', text_coordinate, cv2.FONT_HERSHEY_TRIPLEX,
                    Process.FONT_SIZE, (50, 200, 50), 1, cv2.LINE_AA)

        # Define the two last points
        last_point, point = coordinates[-2], coordinates[-1]

        while True:

            # Draw lines between all the coordinates except the last two
            Board.draw_tracks_lines(self, coordinates, drone_angle)

            start_x, start_y = self.start_x, self.start_y

            cv2.line(self.board, coordinates[-3], coordinates[-2], (0, 255, 0), 5)

            # Right Movements
            if last_point[0] < point[0]:

                p = (last_point[0] + 1, last_point[1])
                cv2.line(self.board, last_point, p, (0, 255, 0), 5)
                last_point = p

            # Left Movements
            elif point[0] < last_point[0]:

                p = (last_point[0] - 1, last_point[1])
                cv2.line(self.board, last_point, p, (0, 255, 0), 5)
                last_point = p

            # Down/Backward Movements
            elif last_point[1] < point[1]:

                p = (last_point[0], last_point[1] + 1)
                cv2.line(self.board, last_point, p, (0, 255, 0), 5)
                last_point = p

            # Up/Forward Movements
            elif point[1] < last_point[1]:

                p = (last_point[0], last_point[1] - 1)
                cv2.line(self.board, last_point, p, (0, 255, 0), 5)
                last_point = p

            else:
                break

            cv2.putText(self.board, f'({((last_point[0] - start_x) / 100)*4},{-((last_point[1] - start_y) / 100)*4})m',
                        (p[0] + 10, p[1] + 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1)

            sleep(0.2)
            self.x_roi = point[0]
            self.y_roi = point[1]
            self.board[p[1]+10:p[1] + 40, p[0]+10:p[0] + 200] = 0
        self.board[2:100, 2:200] = 0

    # Method design board frame and draw lines between previous points
    def draw_tracks_lines(self, coordinates, drone_angle):

        # Fix error
        n = drone_angle
        n += 1

        # Check coordinates array length
        last_point = coordinates[0]

        # Scan all the coordinates and draw it on the board
        for coordinate in coordinates[:-2]:

            # Every 25 cells is 1 meter. add 0.5 to round up / down
            x_coordinate = round((coordinate[0] - self.start_x))
            y_coordinate = round((coordinate[1] - self.start_y))

            # Define the coordinates as a point
            point = ((self.start_x + x_coordinate), int(self.start_y + y_coordinate))

            # Check if the point coordinates is in the board boundaries
            if (0 < point[0] < self.width) or (0 < point[1] < self.height):
                cv2.line(self.board, last_point, point, (0, 255, 0), 5)

            # Update last point
            last_point = point

        pass
