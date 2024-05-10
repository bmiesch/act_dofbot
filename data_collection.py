import cv2
import numpy as np
import os
import json
import time
import threading
from robot import RoboticArm
import rospy
import h5py
from PIL import Image
from training.utils import degrees_to_radians, interpolate_joints


class DataCollection:
    def __init__(self, camera_index, task_name, crop_params=None, camera_name="camera1"):
        self.camera = cv2.VideoCapture(camera_index)
        self.task_dir = f"data/{task_name}"
        self.crop_params = crop_params
        self.camera_name = camera_name
        self.data_buffer = []
        self.robotic_arm = RoboticArm()
        self.running = False
        os.makedirs(self.task_dir, exist_ok=True)

    def save_episode_data(self, episode_id):
        """
        Saves the images and joint angles (data_buffer) for an iteration directly to an HDF5 file.
        """
        hdf5_filename = os.path.join(self.task_dir, f'episode_{episode_id}.hdf5')
        
        with h5py.File(hdf5_filename, 'w') as hdf5_file:
            obs_group = hdf5_file.create_group('observations')
            images_group = obs_group.create_group('images')
            camera_group = images_group.create_group(self.camera_name)
            qpos_dataset = obs_group.create_dataset('qpos', (len(self.data_buffer), 6), dtype='f')
            qvel_dataset = obs_group.create_dataset('qvel', (len(self.data_buffer), 6), dtype='f')  # Placeholder if no velocity data
            action_dataset = obs_group.create_dataset('action', (len(self.data_buffer), 6), dtype='f')

            image_list = []
            for i, (image, joints) in enumerate(self.data_buffer):
                # Convert and save joint angles
                radians_joints = degrees_to_radians(np.array(joints))
                qpos_dataset[i] = radians_joints.astype(np.float32)
                qvel_dataset[i] = np.zeros(6, dtype=np.float32)  # Placeholder if no velocity data

                # Determine the next joints for interpolation
                if i < len(self.data_buffer) - 1:
                    next_joints = self.data_buffer[i + 1][1]
                else:
                    next_joints = joints

                # Calculate current action using interpolation
                current_action = interpolate_joints(joints, next_joints)
                action_dataset[i] = degrees_to_radians(np.array(current_action))

                # Convert image to numpy array and add to list
                image_array = np.array(image)
                image_list.append(image_array)

            # Save all images in a single dataset under the camera name
            camera_group.create_dataset('images', data=np.array(image_list), dtype='uint8')

        self.data_buffer = []

    def episode_data_collection_thread(self):
        """
        Collects images and joint angles from the robotic arm and stores them in the data_buffer.
        """
        self.running = True
        while self.running:
            img = self.capture_image()
            joints_angles = self.robotic_arm.read_joint_angles()
            self.data_buffer.append((img.copy(), joints_angles))
            time.sleep(0.1)

    def start_episode_data_collection(self):
        """
        Starts the episode data collection thread.
        """
        rospy.loginfo("Starting episode data collection")
        self.episode_data_collection_thread_instance = threading.Thread(target=self.episode_data_collection_thread)
        self.episode_data_collection_thread_instance.daemon = True
        self.episode_data_collection_thread_instance.start()

    def stop_episode_data_collection(self, episode_id):
        """
        Stops the episode data collection thread and saves the data for the specified episode.
        """
        rospy.loginfo("Stopping episode data collection")
        self.running = False
        if self.episode_data_collection_thread_instance:
            self.episode_data_collection_thread_instance.join()
        self.save_episode_data(episode_id)

    def capture_image(self):
        """
        Captures an image from the camera and returns it.
        """
        ret, frame = self.camera.read()
        if not ret:
            print("Failed to grab frame")
            return None
        resized_frame = cv2.resize(frame, (548, 274), interpolation=cv2.INTER_AREA)
        if self.crop_params is not None:
            resized_frame = resized_frame[self.crop_params[1]:self.crop_params[1] + self.crop_params[3],
                                          self.crop_params[0]:self.crop_params[0] + self.crop_params[2]]
        return resized_frame

    def calibrate_camera(self):
        """
        Calibrates the camera by selecting a region of interest (ROI) to crop from the image.
        """
        print("Calibrating camera. Press 'q' to quit, 'c' to crop.")

        while True:
            frame = self.capture_image()
            cv2.imshow('Calibration', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                cv2.destroyAllWindows()
                self.crop_params = self.select_fixed_size_roi(frame)
                if self.crop_params is not None:
                    print(f"Crop selected at: {self.crop_params}")
                    break

        cv2.destroyAllWindows()

    def select_fixed_size_roi(self, frame, size=(240, 240)):
        """
        Selects a fixed 240x240 region of interest (ROI) to crop from the image.
        ResNet18 requires a 240x240 input image for training.
        """
        roi_center = (frame.shape[1] // 2, frame.shape[0] // 2)

        def mouse_event(event, x, y, flags, param):
            nonlocal roi_center
            if event == cv2.EVENT_MOUSEMOVE:
                roi_center = (x, y)
                clone = frame.copy()
                cv2.rectangle(clone, (x - size[0] // 2, y - size[1] // 2),
                              (x + size[0] // 2, y + size[1] // 2), (0, 255, 0), 2)
                cv2.imshow("Calibration", clone)

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_event)

        print("Move the box to the desired location and press 'c' to confirm, 'q' to quit.")

        crop_params = None
        while True:
            clone = frame.copy()
            cv2.rectangle(clone, (roi_center[0] - size[0] // 2, roi_center[1] - size[1] // 2),
                          (roi_center[0] + size[0] // 2, roi_center[1] + size[1] // 2), (0, 255, 0), 2)
            cv2.imshow("Calibration", clone)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                x, y = roi_center
                crop_params = (x - size[0] // 2, y - size[1] // 2, size[0], size[1])
                break
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()
        return crop_params
