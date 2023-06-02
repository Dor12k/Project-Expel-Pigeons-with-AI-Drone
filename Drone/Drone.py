

# from _thread import start_new_thread, allocate_lock
# import cv2
import math
import threading
from time import sleep

# import numpy as np

#
# from Path import Path
# from Track import Track
# from Board import Board
# from Motion import Motion
# from Remote import Remote
# from Camera import Camera

# from time import sleep
from djitellopy import Tello

from Drone.Camera import Camera
from Drone.Motion import Motion
from Drone.Path import Path
from Drone.Remote import Remote
from Drone.Track import Track
from Drone.Board import Board
from Process import Process


class Drone:

    start_x = int(Process.WIDTH / 2)
    start_y = int(Process.HEIGHT / 2)

    home = [start_x, start_y]

    # Meter equal 25 cells in array
    meter = 25

    # Patrol for 1,2,3 and 4 meters
    patrol_1m = [(start_x+meter, start_y), (start_x+meter, start_y-meter),
                 (start_x-meter, start_y-meter), (start_x-meter, start_y+meter),
                 (start_x+meter, start_y+meter), (start_x+meter, start_y), (start_x, start_y)]

    patrol_2m = [(start_x+2*meter, start_y), (start_x+2*meter, start_y-2*meter),
                 (start_x-2*meter, start_y-2*meter), (start_x-2*meter, start_y+2*meter),
                 (start_x+2*meter, start_y+2*meter), (start_x+2*meter, start_y), (start_x, start_y)]

    patrol_3m = [(start_x+3*meter, start_y), (start_x+3*meter, start_y-3*meter),
                 (start_x-3*meter, start_y-3*meter), (start_x-3*meter, start_y+3*meter),
                 (start_x+3*meter, start_y+3*meter), (start_x+3*meter, start_y), (start_x, start_y)]

    patrol_4m = [(start_x+4*meter, start_y), (start_x+4*meter, start_y-4*meter),
                 (start_x-4*meter, start_y-4*meter), (start_x-4*meter, start_y+3*meter),
                 (start_x+4*meter, start_y+4*meter), (start_x+4*meter, start_y), (start_x, start_y)]

    # Patrol for 1 meters
    patrol_clockwise_1m = Track()
    patrol_clockwise_1m.add_path(Path("right", 25, 50))
    patrol_clockwise_1m.add_path(Path("backward", 25, 50))
    patrol_clockwise_1m.add_path(Path("left", 25, 100))
    patrol_clockwise_1m.add_path(Path("forward", 25, 100))
    patrol_clockwise_1m.add_path(Path("right", 25, 100))
    patrol_clockwise_1m.add_path(Path("backward", 25, 50))
    patrol_clockwise_1m.add_path(Path("left", 25, 50))

    # Patrol for 2 meters
    patrol_clockwise_2m = Track()
    patrol_clockwise_2m.add_path(Path("right", 25, 100))
    patrol_clockwise_2m.add_path(Path("backward", 25, 100))
    patrol_clockwise_2m.add_path(Path("left", 25, 200))
    patrol_clockwise_2m.add_path(Path("forward", 25, 200))
    patrol_clockwise_2m.add_path(Path("right", 25, 200))
    patrol_clockwise_2m.add_path(Path("backward", 25, 100))
    patrol_clockwise_2m.add_path(Path("left", 25, 100))

    # Patrol for 6 meters
    patrol_clockwise_6m = Track()
    patrol_clockwise_6m.add_path(Path("right", 25, 300))
    patrol_clockwise_6m.add_path(Path("backward", 25, 300))
    patrol_clockwise_6m.add_path(Path("left", 25, 600))
    patrol_clockwise_6m.add_path(Path("forward", 25, 600))
    patrol_clockwise_6m.add_path(Path("right", 25, 600))
    patrol_clockwise_6m.add_path(Path("backward", 25, 300))
    patrol_clockwise_6m.add_path(Path("left", 25, 300))

    # Patrol 2 meters forward and backward
    patrol_fb_2m = Track()
    patrol_fb_2m.add_path(Path("forward", 25, 200))
    patrol_fb_2m.add_path(Path("stand", 1, 0))
    patrol_fb_2m.add_path(Path("backward", 25, 200))

    # Patrol 4 meters forward and backward
    patrol_fb_4m = Track()
    patrol_fb_4m.add_path(Path("forward", 25, 400))
    patrol_fb_4m.add_path(Path("stand", 1, 0))
    patrol_fb_4m.add_path(Path("backward", 25, 400))

    # Patrol 2 meters up and down
    patrol_ud = Track()
    patrol_ud.add_path(Path("up", 50, 200))
    patrol_ud.add_path(Path("stand", 1, 0))
    patrol_ud.add_path(Path("down", 50, 200))

    def __init__(self, drone_sdk_movement_speed, drone_sdk_angular_speed, process):

        # Create Tello object from the SDK
        self.drone = Tello()

        # Remote control by keyboards object
        self.remote = Remote()

        # Track list of Path object (index, speed, distance)
        self.drone_track = Track()

        # Process object to process frames
        self.process = process

        # Create camera object
        self.drone_camera = Camera(self, process.drone_camera)

        # Create black screen to draw the drone track
        self.board = Board(process.board_frame)

        # Define the coordinates of drone home
        self.home = [(self.board.start_x, self.board.start_y)]

        # Board_track is array of track coordinates
        self.board_track = [[self.board.start_x, self.board.start_y]]

        self.location = [self.board.start_x, self.board.start_y]

        self.x_coordinate = self.board.start_x
        self.y_coordinate = self.board.start_y

        # Drone remote status
        self.auto_mode = True

        # Store drone status if tracking or not
        self.on_tracking = False

        # Define the drone start angle and coordinate
        self.drone_angle = 90

        # Counting the distance the drone did
        self.distance_path = 0

        # Counting the distance from the start point (0,0)
        self.distance_from_start = [0, 0]

        # Define the drone motion by speed movement and angular
        self.motion = Motion(drone_sdk_movement_speed, drone_sdk_angular_speed)

        # Define drone angular speed
        self.angular_speed = self.motion.angular_speed

        # Define drone movement speed
        self.movement_speed = self.motion.movement_speed

        # Define angular sdk speed
        self.sdk_angular_speed = drone_sdk_angular_speed

        # Define movement sdk speed
        self.sdk_movement_speed = drone_sdk_movement_speed

    # Send drone bac home
    def back_home(self):
        Drone.track_flight_coordinates(self, self.home)

    # Send drone to patrol
    def patrol(self, patrol_id):

        # Drone patrol 1,2,3 or 4 meter
        if patrol_id == 1:
            Drone.track_flight_coordinates(self, Drone.patrol_1m)
        if patrol_id == 2:
            Drone.track_flight_coordinates(self, Drone.patrol_2m)
        if patrol_id == 3:
            Drone.track_flight_coordinates(self, Drone.patrol_3m)
        if patrol_id == 4:
            Drone.track_flight_coordinates(self, Drone.patrol_4m)

        # Drone patrol 2m forward and backward and land
        if patrol_id == 5:
            Drone.track_flight(self, Drone.patrol_fb_2m)
        if patrol_id == 6:
            Drone.track_flight(self, Drone.patrol_fb_4m)
        if patrol_id == 7:
            Drone.track_flight(self, Drone.patrol_clockwise_1m)
        if patrol_id == 8:
            Drone.track_flight(self, Drone.patrol_clockwise_2m)
        if patrol_id == 9:
            Drone.track_flight(self, Drone.patrol_clockwise_6m)

        if patrol_id == 0:
            Drone.track_flight(self, Drone.patrol_ud)

        self.on_tracking = False

        return

    # Return x,y distance from given coordinate
    def get_distance(self, point):
        a = (point[0] - self.x_coordinate)
        b = (point[1] - self.y_coordinate)
        distance = (a, b)
        return distance

    # Get distance line from (x, y) coordinate
    def get_destination(self, point):
        distance = math.dist(self.location, (point[0], point[1]))
        return distance

    # Flight to (x, y) coordinate
    def arrive_destination(self, point, from_keyboard):

        # Define ethe type command
        if from_keyboard == 1:
            # Keyboards
            type_command = 1
        else:
            # Track
            type_command = 0

        # Get (x, y) distance from point
        paths = self.get_distance(point)

        # Define the x, y distance from point
        x, y = paths[0], paths[1]

        # Move to the sides
        if x < 0:
            Drone.left(self, -x, type_command)
        elif 0 < x:
            Drone.right(self, x, type_command)

        # Stand
        Drone.stand(self)

        # Move forward/backward
        if y < 0:
            Drone.backward(self, y, type_command)
        elif 0 < y:
            Drone.forward(self, -y, type_command)

    # Drone flight by coordinates track
    def track_flight_coordinates(self, coordinates):

        # Define it from_keyboard to draw the track
        from_keyboard = 0

        # Arrive to next point every time
        for coordinate in coordinates:
            # if user press on a keyboard the track stop
            if self.remote.ev.is_set():
                # sleep(3)
                break
            Drone.arrive_destination(self, coordinate, from_keyboard)

        # Reset the event variable
        self.remote.ev.clear()

    # Get Track object which is array of Path objects
    def track_flight(self, track):

        # Define ethe type command
        from_track = 2

        # Movement every path in the track
        for i, path in enumerate(track.track):

            # if user press on a keyboard the track stop
            if self.remote.ev.is_set():
                print("break")
                break

            # Print our path
            print("i: ", path.__str__())

            # Extract the values from the Path object
            path_speed = path.speed
            path_distance = path.distance
            path_direction = path.direction

            # Update the drone speed SDK
            self.sdk_movement_speed = path_speed

            if path_direction == "stand":
                Drone.stand(self)

            elif path_direction == "right":
                Drone.right(self, path_distance, from_track)
            elif path_direction == "left":
                Drone.left(self, path_distance, from_track)
            elif path_direction == "forward":
                Drone.forward(self, path_distance, from_track)
            elif path_direction == "backward":
                Drone.backward(self, path_distance, from_track)
            elif path_direction == "up":
                Drone.up(self, path_distance, from_track)
            elif path_direction == "down":
                Drone.down(self, path_distance, from_track)
            elif path_direction == "rotate left":
                degree = path_distance
                Drone.rotate_right(self, degree)
                print("Rotate Right: ", degree, "degree")
            elif path_direction == "rotate right":
                degree = path_distance
                Drone.rotate_left(self, degree)
                print("Rotate Left: ", degree, "degree")

        # Reset the event variable
        self.remote.ev.clear()

    # Move forward
    def forward(self, distance, type_command):

        # Aim the proportion to the board
        if type_command == 0:
            distance = distance * 4

        # Make the movement from Motion class
        tf = threading.Thread(target=self.motion.forward, args=(self.drone, distance))
        tf.name = "Drone: forward"
        tf.start()

        # Update drone flight distance
        self.distance_path += distance

        # Update the distance from the start point
        self.distance_from_start[1] += distance

        # Create movement object from Path class (direction, speed, distance)
        movement = Path("forward", self.sdk_movement_speed, distance)

        # Add the path to the track the drone did
        self.drone_track.add_path(movement)

        # Back to original distance
        if type_command == 0:
            distance = distance / 4

        # Update drone coordinates
        d_x = round(distance * math.cos(math.radians(self.drone_angle)))
        d_y = round(distance * math.sin(math.radians(self.drone_angle)))
        if type_command == 1 or type_command == 2:
            d_x = round(d_x / 4)
            d_y = round(d_y / 4)

        # Update drone coordinates location
        self.location = [self.x_coordinate + d_x, self.y_coordinate - d_y]

        # Draw the track in the drone board
        # if type_command = 0 is mean the command came from board track
        # if type_command = 1 is mean the command came from keyboard
        # if type_command = 2 is mean the command came from drone track
        Drone.draw_track(self, type_command)

        tf.join()
        return

    # Move backward
    def backward(self, distance, type_command):

        # Aim the proportion to the board
        if type_command == 0:
            distance = distance * 4

        # Make the movement from Motion class
        tb = threading.Thread(target=self.motion.backward, args=(self.drone, distance))
        tb.name = "Drone: Backward"
        tb.start()

        # Update drone flight distance
        self.distance_path += distance

        # Update the distance from the start point
        self.distance_from_start[1] -= distance

        # Back to original distance
        if type_command == 0:
            distance = distance / 4

        # Update drone coordinate
        d_x = round(distance * math.cos(math.radians(self.drone_angle)))
        d_y = round(distance * math.sin(math.radians(self.drone_angle)))
        if type_command == 1 or type_command == 2:
            d_x = round(d_x / 4)
            d_y = round(d_y / 4)

        # Update drone location
        self.location = [self.x_coordinate - d_x, self.y_coordinate + d_y]

        # Create movement object from Path class (direction, speed, distance)
        movement = Path("backward", self.sdk_movement_speed, distance)

        # Add path to the drone track
        self.drone_track.add_path(movement)

        # Draw the track in the drone board
        # if type_command = 0 is mean the command came from board track
        # if type_command = 1 is mean the command came from keyboard
        # if type_command = 2 is mean the command came from drone track
        Drone.draw_track(self, type_command)

        tb.join()
        return

    # Move right
    def right(self, distance, type_command):

        # Aim the proportion to the board
        if type_command == 0:
            distance = distance * 4

        # Make the movement from Motion class
        tr = threading.Thread(target=self.motion.right, args=(self.drone, distance))
        tr.name = "Drone: right"
        tr.start()

        # Update drone flight distance
        self.distance_path += distance

        # Update the distance from the start point
        self.distance_from_start[0] += distance

        # Back to original distance
        if type_command == 0:
            distance = distance / 4

        # Update drone coordinate
        d_x = round(distance * math.cos(math.radians(self.drone_angle-90)))
        d_y = round(distance * math.sin(math.radians(self.drone_angle-90)))
        if type_command == 1 or type_command == 2:
            d_x = round(d_x / 4)
            d_y = round(d_y / 4)

        # Update drone location
        self.location = [self.x_coordinate + d_x, self.y_coordinate - d_y]

        # Create movement object from Path class (direction, speed, distance)
        movement = Path("right", self.sdk_movement_speed, distance)
        self.drone_track.add_path(movement)

        # Draw the track in the drone board
        # if type_command = 0 is mean the command came from board track
        # if type_command = 1 is mean the command came from keyboard
        # if type_command = 2 is mean the command came from drone track
        Drone.draw_track(self, type_command)

        tr.join()
        return

    # Move left
    def left(self, distance, type_command):

        # Aim the proportion to the board
        if type_command == 0:
            distance = distance * 4

        # Make the movement from Motion class
        tl = threading.Thread(target=self.motion.left, args=(self.drone, distance))
        tl.name = "Drone: left"
        tl.start()

        # Update drone flight distance
        self.distance_path += distance

        # Update the distance from the start point
        self.distance_from_start[0] -= distance

        # Back to original distance
        if type_command == 0:
            distance = distance / 4

        # Update drone coordinate
        d_x = round(distance * math.cos(math.radians(self.drone_angle+90)))
        d_y = round(distance * math.sin(math.radians(self.drone_angle+90)))
        if type_command == 1 or type_command == 2:
            d_x = round(d_x / 4)
            d_y = round(d_y / 4)

        # Update drone location
        self.location = [self.x_coordinate + d_x, self.y_coordinate - d_y]

        # Create movement object from Path class (direction, speed, distance)
        movement = Path("left", self.sdk_movement_speed, distance)

        # Add path to drone track
        self.drone_track.add_path(movement)

        # Draw the track in the drone board
        # if type_command = 0 is mean the command came from board track
        # if type_command = 1 is mean the command came from keyboard
        # if type_command = 2 is mean the command came from drone track
        Drone.draw_track(self, type_command)

        tl.join()
        return

    # Draw drone board track on the board
    def draw_track(self, type_command):

        # Draw the track in the drone board
        # if type_command = 0 is mean the command came from board track
        # if type_command = 1 is mean the command came from keyboard
        # if type_command = 2 is mean the command came from drone track
        if type_command == 0:
            self.board.draw_tracks(self.board_track, self.drone_angle)
        if type_command == 1:
            self.board.draw_points(self.board_track, self.drone_angle)
        if type_command == 2:
            self.board.draw_tracks(self.board_track, self.drone_angle)

    # Move up
    def up(self, distance, type_command):

        # Draw the track in the drone board
        # if type_command = 0 is mean the command came from board track
        # if type_command = 1 is mean the command came from keyboard
        # if type_command = 2 is mean the command came from drone track
        tu = threading.Thread(target=self.motion.up, args=(self.drone, distance))
        tu.name = "Drone: up"
        tu.start()

        # Create movement object from Path class (direction, speed, distance)
        movement = Path("up", self.sdk_movement_speed, distance)
        self.drone_track.add_path(movement)

        tu.join()
        return

    # Move down
    def down(self, distance, type_command):

        # Draw the track in the drone board
        # if type_command = 0 is mean the command came from board track
        # if type_command = 1 is mean the command came from keyboard
        # if type_command = 2 is mean the command came from drone track

        # Make the movement from Motion class
        td = threading.Thread(target=self.motion.down, args=(self.drone, distance))
        td.name = "Drone: down"
        td.start()

        # Create movement object from Path class (direction, speed, distance)
        movement = Path("down", self.sdk_movement_speed, distance)
        self.drone_track.add_path(movement)

        td.join()
        return

    # Drone takeoff
    def takeoff(self):
        print("Drone takeoff..")
        self.motion.takeoff(self.drone)
        print("Ready to flight")

    # Drone land
    def land(self):
        print("Drone Landing..")
        self.motion.land(self.drone)
        print("Drona landed")

    # Stand in place
    def stand(self):
        # Make the movement from Motion class
        self.motion.stand(self.drone)

        # Create movement object from Path class (direction, speed, distance)
        movement = Path("stand", self.sdk_movement_speed, 0)
        self.drone_track.add_path(movement)

    # Rotate drone X degree to left
    def rotate_left(self, angle):

        # Update drone angle
        # self.__drone_angle(angle)
        self.drone_angle = self.drone_angle + angle

        # Make the movement from Motion class
        self.motion.rotate_left(self.drone, angle)

        # Create movement object from Path class (direction, speed, distance)
        movement = Path("rotate left", self.sdk_movement_speed, angle)
        self.drone_track.add_path(movement)

    # Rotate drone X degree to right
    def rotate_right(self, angle):

        # Update drone angle
        self.drone_angle = self.drone_angle - angle

        # Make the movement from Motion class
        self.motion.rotate_right(self.drone, angle)

        # Create movement object from Path class (direction, speed, distance)
        movement = Path("rotate right", -self.sdk_movement_speed, angle)
        self.drone_track.add_path(movement)

    # Method translate keyboards input from user
    def remote_control(self, pressed_key, detection_obj):
        system_on = self.remote.get_keyboard_input(self, pressed_key, detection_obj)
        return system_on

    @property
    def drone_camera(self):
        return self.__drone_camera

    @drone_camera.setter
    def drone_camera(self, camera):
        self.__drone_camera = camera

    # Connect to drone camera
    def stream_on(self):
        self.drone_camera.stream_on()

    # Disconnect from drone camera
    def stream_off(self):
        self.drone_camera.stream_off()

    # Read frames from drone camera
    def read_frame_from_camera(self):
        drone_frame, drone_mask = self.drone_camera.read_frame_from_camera()

        return drone_frame, drone_mask

    @property
    def motion(self):
        return self.__motion

    @motion.setter
    def motion(self, drone_motion):
        self.__motion = drone_motion

    @property
    def movement_speed(self):
        return self.__movement_speed

    @movement_speed.setter
    def movement_speed(self, drone_movement_speed):
        self.__movement_speed = drone_movement_speed

    @property
    def angular_speed(self):
        return self.__angular_speed

    @angular_speed.setter
    def angular_speed(self, drone_angular_speed):
        self.__angular_speed = drone_angular_speed

    @property
    def sdk_angular_speed(self):
        return self.__sdk_angular_speed

    @sdk_angular_speed.setter
    def sdk_angular_speed(self, drone_sdk_angular_speed):
        if 100 < drone_sdk_angular_speed or drone_sdk_angular_speed < -100:
            print("Class Drone: sdk speed must be between -100~100")
            # self.__sdk_angular_speed = 0
        else:
            self.__sdk_angular_speed = drone_sdk_angular_speed
            self.__motion.sdk_angular_speed = drone_sdk_angular_speed
            self.__angular_speed = self.motion.angular_speed

    @property
    def sdk_movement_speed(self):
        return self.__sdk_movement_speed

    @sdk_movement_speed.setter
    def sdk_movement_speed(self, drone_sdk_movement_speed):
        if 100 < drone_sdk_movement_speed or drone_sdk_movement_speed < -100:
            print("Class Drone: sdk speed must be between -100~100. value: ", drone_sdk_movement_speed)
            # self.__sdk_movement_speed = 0
        else:
            self.__sdk_movement_speed = drone_sdk_movement_speed
            self.__motion.sdk_movement_speed = drone_sdk_movement_speed
            self.__movement_speed = self.motion.movement_speed

    # Return int drone battery
    def get_battery(self):
        return self.drone.get_battery()

    # Manage drone battery
    def drone_battery_management(self):
        # Store the drone battery value
        drone_battery = self.drone.get_battery()

        # Print battery evert 10%
        if drone_battery % 10 == 0:
            print("Battery: ", self.drone.get_battery(), "%")

        # Safe land when battery is low as 5%
        if drone_battery == 5:
            # Reduce drone speed to 0 before is landing
            print("**WARNING** Battery: ", self.drone.get_battery(), "%", " landing")
            Drone.land(self)

    # Connect to drone
    def connect(self):
        self.drone.connect()

    @property
    def drone(self):
        return self.__drone

    @drone.setter
    def drone(self, tello):
        self.__drone = tello

    # Return drone angle
    @property
    def drone_angle(self):
        return self.__drone_angle

    # Set drone angle
    @drone_angle.setter
    def drone_angle(self, angle):

        self.__drone_angle = angle

        if 360 < self.__drone_angle:
            self.__drone_angle = float(self.__drone_angle - 360)
        if self.__drone_angle < -360:
            self.__drone_angle = float(self.__drone_angle + 360)

        # self.motion.set_angular(angle)

    # Get the track the drone did as array of Path objects
    @property
    def drone_track(self):
        # Drone track is a Track object
        return self.__drone_track

    # Set the track the drone did
    @drone_track.setter
    def drone_track(self, track):
        # track should be Track object
        self.__drone_track = track

    # Return the track the drone did as array of coordinates
    @property
    def board_track(self):
        return self.__board_track

    # Set the track drone did as array od coordinates
    @board_track.setter
    def board_track(self, track):
        self.__board_track = track

    # Return drone board which is numpy array
    @property
    def board(self):
        return self.__board

    @board.setter
    def board(self, drone_board):
        self.__board = drone_board

    # Return the path distance
    @property
    def distance_path(self):
        return self.__distance_path

    # Update the path distance
    @distance_path.setter
    def distance_path(self, distance):
        if distance == 0:
            self.__distance_path = distance
        else:
            self.__distance_path = self.distance_path + distance

    # Get distance from start point
    @property
    def distance_from_start(self):
        return self.__distance_from_start

    # Update the distance from start point
    @distance_from_start.setter
    def distance_from_start(self, distance):
        self.__distance_from_start = distance

    # Get drone coordinate location as point (x, y)
    @property
    def location(self):
        return self.__location

    # Set drone coordinate location
    @location.setter
    def location(self, coordinate):

        self.__location = [coordinate[0], coordinate[1]]

        self.x_coordinate, self.y_coordinate = self.location[0], self.location[1]

        # Add current coordinate to the board track array
        self.board_track.append(self.location)

    # Get x coordinate
    @property
    def x_coordinate(self):
        return self.__x_coordinate

    # Set x coordinate
    @x_coordinate.setter
    def x_coordinate(self, x):
        self.__x_coordinate = x
        self.location[0] = self.__x_coordinate

        # Add current coordinate to the board track array
        # self.board_track.append(self.location)

    # Get y coordinate
    @property
    def y_coordinate(self):
        return self.__y_coordinate

    # Set y coordinate
    @y_coordinate.setter
    def y_coordinate(self, y):
        self.__y_coordinate = y
        self.location[1] = self.__y_coordinate

        # Add current coordinate to the board track array
        # self.board_track.append(self.location)