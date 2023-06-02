
import cv2
import time
import threading
# from time import sleep

from _thread import start_new_thread  # , allocate_lock


i = 0


class Remote:

    # Thread event to change Drone object status of tracking
    ev = threading.Event()

    # Constant ASCII code for keyboards
    FORWARD, BACKWARD, LEFT, RIGHT, STAND = 2490368, 2621440, 2424832, 2555904, 32

    def __init__(self):
        self.temp = 0

    @staticmethod
    def get_keyboard_input(drone, pressed_key, detection_obj):

        # Variable define if to focus or not
        global i
        i = i + 1

        # Variable keep or brake system look in main class
        system_on = True

        # Represent the time to flight for one keyboard tapping
        interval = 0.5

        # Send to function a sign the movement came from the keyboard
        from_keyboard = 1

        # Store the drone SDK speed
        speed = drone.sdk_movement_speed
        angle = drone.sdk_angular_speed

        # Represent the distance drone flight for one keyboard tapping
        angular_interval = round(drone.motion.angular_speed * interval)
        distance_interval = round(drone.motion.movement_speed * interval)

        # Arrow Up key - Forward movements
        if pressed_key == Remote.FORWARD:
            if drone.on_tracking:
                Remote.ev.set()

            # Send command to flight forward
            # drone.forward(distance_interval, from_keyboard)
            tf = threading.Thread(target=drone.forward, args=(distance_interval, from_keyboard))
            tf.name = "Remote: forward " + str(i)
            tf.start()
            # t1.join()

        # Arrow Down key - Backward movements
        elif pressed_key == Remote.BACKWARD:
            if drone.on_tracking:
                Remote.ev.set()

            # Send command to flight forward
            # drone.backward(distance_interval, from_keyboard)
            tb = threading.Thread(target=drone.backward, args=(distance_interval, from_keyboard))
            tb.name = "Remote: backward " + str(i)
            tb.start()

        # Arrow Left key - Flight left
        elif pressed_key == Remote.LEFT:
            if drone.on_tracking:
                Remote.ev.set()

            # Send command to flight forward
            # drone.left(distance_interval, from_keyboard)
            tl = threading.Thread(target=drone.left, args=(distance_interval, from_keyboard))
            tl.name = "Remote: left " + str(i)
            tl.start()

        # Arrow Right key - Flight right
        elif pressed_key == Remote.RIGHT:
            if drone.on_tracking:
                Remote.ev.set()

            # Send command to flight forward
            # drone.right(distance_interval, from_keyboard)
            tr = threading.Thread(target=drone.right, args=(distance_interval, from_keyboard))
            tr.name = "Remote: right " + str(i)
            tr.start()

        # Space key - Stand
        elif pressed_key == Remote.STAND:
            ts = threading.Thread(target=drone.stand, args=())
            ts.name = "Remote: stand " + str(i)
            ts.start()

        # W key - Flight up
        elif pressed_key == ord("w"):
            if drone.on_tracking:
                Remote.ev.set()

            # drone.up(from_keyboard)
            tu = threading.Thread(target=drone.up, args=(distance_interval, from_keyboard))
            tu.name = "Remote: up " + str(i)
            tu.start()

        # S key - Flight down
        elif pressed_key == ord("s"):
            if drone.on_tracking:
                Remote.ev.set()

            # drone.down(from_keyboard)
            td = threading.Thread(target=drone.down, args=(distance_interval, from_keyboard))
            td.name = "Remote: down " + str(i)
            td.start()

        # A key - Rotate left
        elif pressed_key == ord("a"):
            # Drone rotate to right
            drone.rotate_left(angular_interval)

        # D key - Rotate right
        if pressed_key == ord("d"):
            # Drone rotate to right
            drone.rotate_right(angular_interval)

        # E key - Drone Takeoff
        if pressed_key == ord("e"):
            drone.takeoff()
            pass

        # Q key - Drone Land
        if pressed_key == ord("q"):
            drone.land()
            pass

        # C key = turn on/off camera
        if pressed_key == ord('c'):
            # If camera true then turn it off. else turn it on
            if drone.drone_camera.camera:
                drone.stream_off()
                drone.drone_camera.camera = False
            else:
                drone.stream_on()
                drone.drone_camera.camera = True

        # Save image from the drone camera
        if pressed_key == ord("z"):
            img = drone.drone_camera.camera_frame
            cv2.imwrite(f'Resources/Images/{time.time()}.jpg', img)
            time.sleep(0.5)

        # End the program
        if pressed_key == ord('o'):
            drone.on_tracking = False
            # drone.land()
            system_on = False
            return system_on

        # If 'n' is pressed, we catch's the frame and define it as the background
        if pressed_key == ord('n'):
            detection_obj.tracking_on = False
            pass

        # Send drone to patrol according to the id
        if pressed_key == ord('1'):
            drone.on_tracking = True
            start_new_thread(drone.patrol, (1, ))
        if pressed_key == ord('2'):
            drone.on_tracking = True
            start_new_thread(drone.patrol, (2, ))
        if pressed_key == ord('3'):
            drone.on_tracking = True
            start_new_thread(drone.patrol, (3, ))
        if pressed_key == ord('4'):
            drone.on_tracking = True
            start_new_thread(drone.patrol, (4, ))

        # Drone patrol forward and backward
        if pressed_key == ord('5'):
            drone.on_tracking = True
            start_new_thread(drone.patrol, (5, ))
        if pressed_key == ord('6'):
            drone.on_tracking = True
            start_new_thread(drone.patrol, (6, ))
        if pressed_key == ord('7'):
            drone.on_tracking = True
            start_new_thread(drone.patrol, (7, ))
        if pressed_key == ord('8'):
            drone.on_tracking = True
            start_new_thread(drone.patrol, (8, ))
        if pressed_key == ord('9'):
            drone.on_tracking = True
            start_new_thread(drone.patrol, (9, ))

        # Drone patrol up and down
        if pressed_key == ord('0'):
            drone.on_tracking = True
            start_new_thread(drone.patrol, (0, ))

        # Send drone back to home (start drone coordinates)
        if pressed_key == ord('h'):
            drone.on_tracking = True
            start_new_thread(drone.back_home, ())

        # Drone take off and patrol forward and backward
        if pressed_key == ord('t'):
            drone.on_tracking = True
            drone.takeoff()
            start_new_thread(drone.patrol_fb, (6, ))
            pass

        # Change drone mode: automatic/user control
        if pressed_key == ord('m'):
            if drone.auto_mode:
                drone.auto_mode = False
            else:
                drone.auto_mode = True

        # Drawing target on the screen
        if pressed_key == ord('f'):
            drone.on_tracking = True
            if drone.drone_camera.target == 0:
                drone.drone_camera.target = 1
            elif drone.drone_camera.target == 1:
                drone.drone_camera.target = 0

        # Resent drone tracking
        if pressed_key == ord('r'):
            # drone.on_tracking = False
            drone.drone_camera.tracking = False
            pass

        # Switch drone tracking on and off
        if pressed_key == ord('l'):
            if drone.drone_camera.tracking_mode:
                drone.drone_camera.tracking_mode = False
            else:
                drone.drone_camera.tracking_mode = True

        # Switch mode test
        if pressed_key == ord('k'):
            if drone.drone_camera.on_test:
                drone.drone_camera.on_test = False
            else:
                drone.drone_camera.on_test = True
        else:
            pass

        return system_on
