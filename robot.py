import time
from Arm_Lib import Arm_Device


class RoboticArm:
    def __init__(self):
        self.Arm = Arm_Device()
        time.sleep(1)
        self.set_home_position()

    def move_servos(self, positions, duration_ms):
        """
        Moves servos to specified positions over a given duration.

        :param positions: List of servo positions.
        :param duration_ms: Duration in milliseconds for the servo movement.
        """
        try:
            self.Arm.Arm_serial_servo_write6_array(positions, duration_ms)
            time.sleep(duration_ms / 1000.0)
        except Exception as e:
            print(f"Failed to move servos: {e}")

    def move_single_servo(self, servo_id, position, duration_ms):
        """
        Moves a single servo to a specified position over a given duration.

        :param servo_id: ID of the servo to move.
        :param position: Position to move the servo to.
        :param duration_ms: Duration in milliseconds for the servo movement.
        """
        try:
            self.Arm.Arm_serial_servo_write(servo_id, position, duration_ms)
            time.sleep(duration_ms / 1000.0)
        except Exception as e:
            print(f"Failed to move servo: {e}")
    
    def set_home_position(self):
        """
        Sets the robotic arm to its home position.
        """
        home_positions = [90, 130, 0, 0, 90, 30]
        duration_ms = 1000
        self.move_servos(home_positions, duration_ms)
        print("Robot has been set to its home position.")

    def set_custom_position(self, positions, duration_ms=500):
        """
        Sets the robotic arm to a custom position.

        :param positions: List of servo positions to set the arm to.
        """
        self.move_servos(positions, duration_ms)
        time.sleep(duration_ms / 1000.0)
        print("Robot has been set to a custom position.")

    def read_joint_angles(self):
        """
        Returns the current joint positions of the robotic arm, with retries for null values.
        """
        max_retries = 5  # Maximum number of retries per servo
        joint_positions = []
        for i in range(6):
            retries = 0
            while retries < max_retries:
                position = self.Arm.Arm_serial_servo_read(i + 1)
                if position is not None:
                    joint_positions.append(position)
                    break
                retries += 1
            if retries == max_retries:
                raise ValueError(f"Failed to read position for servo {i+1} after {max_retries} retries")
        return joint_positions
