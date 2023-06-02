
# Class object represent one path in the drone track
class Path:

    # Drone speed must be between -100 ~ 100
    min_speed, max_speed = -100, 100

    # Drone directions translate to index numbers of SDK rc_command
    LEFT_RIGHT, FORWARD_BACKWARD, UP_DOWN, ROTATE_LEFT_ROTATE_RIGHT = 0, 1, 2, 3

    def __init__(self, direction, drone_speed, distance):

        # Distance - movement in cm
        # Speed - the SDK value in the Tello.rc command
        # Direction - index in rc command send_rc_control(0, 0, 0, 0)
        self.direction = direction
        self.speed = drone_speed
        self.distance = distance

    @property
    # Return the speed value in rc command send_rc_control(0, 0, 0, 0)
    def speed(self):
        return self.__speed

    @speed.setter
    # Set the speed movement
    def speed(self, drone_speed):
        if (Path.max_speed < drone_speed) or (drone_speed < Path.min_speed):
            print("Speed must be between -100~100")
            self.__speed = 0
        else:
            self.__speed = drone_speed

    # Return distance on movement in cm
    @property
    def distance(self):
        return self.__distance

    # Set the movement distance in cm
    @distance.setter
    def distance(self, path_distance):
        self.__distance = path_distance

    # Return index in rc command send_rc_control(0, 0, 0, 0)
    @property
    def direction(self):
        return self.__direction

    # Set index in rc command send_rc_control(0, 0, 0, 0)
    @direction.setter
    def direction(self, path_direction):

        if path_direction == "right":
            self.__direction = "right"
        elif path_direction == "left":
            self.__direction = "left"

        elif path_direction == "forward":
            self.__direction = "forward"
        elif path_direction == "backward":
            self.__direction = "backward"

        elif path_direction == "up":
            self.__direction = "up"
        elif path_direction == "down":
            self.__direction = "down"

        elif path_direction == "rotate right":
            self.__direction = "rotate right"
        elif path_direction == "rotate left":
            self.__direction = "rotate left"

        elif path_direction == "stand":
            self.__direction = "stand"
        else:
            print("CLASS: Path - Invalid direction value")
            self.__direction = "stand"

    def get_sdk_path(self):

        speed = self.speed
        # distance = self.distance
        direction = self.direction

        if direction == "right":
            return speed, 0, 0, 0
        if direction == "left":
            return -speed, 0, 0, 0

        if direction == "forward":
            return 0, speed, 0, 0
        if direction == "backward":
            return 0, -speed, 0, 0

        if direction == "up":
            return 0, 0, speed, 0
        if direction == "down":
            return 0, 0, -speed, 0

        if direction == "rotate_right":
            return 0, 0, 0, speed
        if direction == "rotate_left":
            return 0, 0, 0, -speed
        else:
            print("CLASS: Path - Invalid direction value")
            return 0, 0, 0, 0

    # Return "Path(direction, speed, distance)"
    def __repr__(self):
        return f"Path({self.__direction},{self.__speed},{self.__distance})"

    # Return "Path(direction: direction, speed: speed, distance: distance)"
    def __str__(self):
        return f"Direction: {self.__direction}, Speed: {self.__speed}, Distance: {self.__distance}"
