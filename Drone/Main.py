
import cv2
import numpy as np
# import threading
# from Remote import Remote
from time import sleep

from djitellopy import Tello

from Drone import Drone
from timeit import default_timer as timer

drone = Drone(25, 10)

# drone.connect()

is_track = 0


# Function management the main window
def display_window(tracking, drone_obj):

    drone_frame, drone_mask = drone_obj.read_frame_from_camera()

    # # Camera variable connecting to drone camera
    # camera = drone.drone_camera
    #
    # # Function return a frame from drone camera
    # frame = camera.read_frame()

    # Adding some features to frame window
    # drone_cam, battery = camera.design_frame(frame)
    drone_cam, battery = drone_obj.drone_camera.design_frame(drone_frame)

    # Variable store the board that drawing the drone movements
    drone_board = drone_obj.board.board

    # Collecting the windows to one main window
    main_window = np.hstack((drone_mask, drone_cam, drone_board))

    # Display the main window
    cv2.imshow("main_window", main_window)

    # Cleaning some pixels for the next iterate
    if tracking == 0:
        point = (drone_obj.board.x_roi, drone_obj.board.y_roi)
        drone_obj.board.board[point[1] + 10:point[1] + 40, point[0]+10:point[0] + 210] = 0

    return drone_cam, drone_board, battery


start_timer = timer()
# drone.drone.set_video_direction(Tello.CAMERA_DOWNWARD)
while True:

    # Store key pressed from user
    key = cv2.waitKeyEx(is_track)

    stop_timer = timer()
    t = stop_timer - start_timer
    # print("T: ", t)
    if 60 * 1 < t:
        print("Timer: ", t)
        # print("Battery: ", drone.get_battery())
        # drone.drone.send_keepalive()
        # drone.stand()
        # print("Height: ", drone.drone.get_height())
        # print("Distance TOF: ", drone.drone.get_distance_tof())
        # drone.drone.send_command_with_return()
        # drone.drone.send_rc_control(0, 0, 0, 1)
        # drone.drone.turn_motor_on()
        # sleep(3)
        # drone.drone.turn_motor_off()
        # sleep(3)
        # drone.drone.initiate_throw_takeoff()
        # drone.drone.move("forward", 0)
        start_timer = timer()
        # key = ord('z')

    # print("key: ", key, " is track: ", is_track)

    # If key -1 mean none keyboard pressed so do nothing
    if not (key == -1 and is_track == 1):

        # stop, is_track = Remote.get_keyboard_input(remote, key, is_track)
        stop, is_track = drone.remote_control(key, is_track)
        if stop:
            break

    # Function display frames on the screen
    img, board, drone_battery = display_window(is_track, drone)


print("Timer: ", t)
# print("Battery: ", drone.get_battery())
# drone.drone.streamoff()
# print(str(remote.drone.drone_track))
# print("Drone coordinates track: ", drone.board_track)
drone.drone.end()
cv2.destroyAllWindows()
