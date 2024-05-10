import cv2 as cv
import threading
from time import sleep
import rospy
from pick_and_place import PickAndPlace
from data_collection import DataCollection

exit_flag = False


def process_image(capture, game, color_hsv):
    """Capture and process an image from the camera."""
    _, img = capture.read()
    img = cv.resize(img, (640, 480))
    if game.dp:
        img = game.calibration.Perspective_transform(game.dp, img)
    cv.imshow("Camera Feed", img)
    return img

def handle_block_detection(game, img, color_hsv, data_collection, task_iteration):
    """Detect and handle block movement based on detected position."""
    block_position = game.get_block_position(img, color_hsv)
    if block_position:
        joints = game.calculate_inverse_kinematics(block_position)
        if joints:
            game.move_block(joints, data_collection, task_iteration)
            return True
    return False

def game_loop(game, color_hsv, data_collection):
    global exit_flag
    rospy.on_shutdown(shutdown_hook)
    capture = cv.VideoCapture(0) # Camera on top of dofbot
    task_iteration = 0

    while capture.isOpened() and not exit_flag and not rospy.is_shutdown():
        img = process_image(capture, game, color_hsv)
        if handle_block_detection(game, img, color_hsv, data_collection, task_iteration):
            task_iteration += 1
            sleep(3)  # Delay after moving a block
        else:
            rospy.loginfo("No block detected")

        if cv.waitKey(1) & 0xFF == ord('q'):
            exit_flag = True

        rospy.loginfo("Iteration completed")
        sleep(0.1)

    capture.release()

def shutdown_hook():
    """Handle shutdown of ROS node."""
    global exit_flag
    exit_flag = True
    print("ROS node is shutting down")

def main():
    game = PickAndPlace()
    color_hsv = [(35, 43, 46), (77, 255, 255)]
    data_collection = DataCollection(camera_index=1, task_name="pick_and_place")
    data_collection.calibrate_camera()

    game_thread = threading.Thread(target=game_loop, args=(game, color_hsv, data_collection))
    game_thread.start()

    try:
        game_thread.join()
    except KeyboardInterrupt:
        global exit_flag
        exit_flag = True
        print("Interrupted by user")
        game_thread.join()
    finally:
        data_collection.stop_task_data_collection(-1)
        print("Data collection stopped in finally")

if __name__ == '__main__':
    main()