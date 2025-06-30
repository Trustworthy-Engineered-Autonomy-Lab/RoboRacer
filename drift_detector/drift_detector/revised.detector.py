import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from vesc_msgs.msg import VescImuStamped
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String, Float64
import numpy as np
from queue import Queue
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class DriftDetector(Node):
    def __init__(self):
        super().__init__('drifting_detector')
        
        self.wheelbase = 0.32  # 32 cm in meters
        self.tirewidth = 0.04445
        self.mass = 3.333
        self.gravity = 9.81
        self.force_of_gravity = self.mass * self.gravity

        self.mus = []

        self.ackermann_subscriber = self.create_subscription(
            AckermannDriveStamped,
            '/ackermann_cmd',
            self.ackermann_callback,
            10
        )
        self.imu_subscriber = self.create_subscription(
            Imu,
            '/sensors/imu/raw',
            self.imu_callback,
            10
        )
        self.odomfil_subscriber = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.odom_callback,
            10
        )

        self.drifting_publisher = self.create_publisher(Bool, 'is_drifting', 10)

        self.throttle = float('inf')
        self.steering_angle = float('inf')
        self.turning_radius = float('inf')
        self.theor_ang_vel = float('inf')
        self.twist_angular_z = float('inf')
        self.linear_acceleration = float('inf')
        self.y_lin_accel = float('inf')
        self.timestamp = 0.0

        self.drifting = False
        self.angular = False
        self.linear = False
        self.drifting_timestamp = 0.0

    def ackermann_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        throttle = msg.drive.speed
        steering_angle = msg.drive.steering_angle
        turning_radius = self.wheelbase / np.tan(steering_angle) if steering_angle != 0 else float('inf')
        
        self.timestamp = timestamp
        self.throttle = throttle
        self.steering_angle = steering_angle
        self.turning_radius = turning_radius
        self.theor_ang_vel = throttle / turning_radius if turning_radius != 0 else float('inf')

        self.check_drifting()

    def imu_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        angular_velocity = msg.angular_velocity.z
        linear_acceleration = msg.linear_acceleration.x
        y_lin_accel = msg.linear_acceleration.y

        self.timestamp = timestamp
        self.twist_angular_z = angular_velocity
        self.linear_acceleration = linear_acceleration
        self.y_lin_accel = y_lin_accel

        self.check_drifting()

    def odom_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        odom_linear_x = msg.twist.twist.linear.x
        odom_linear_y = msg.twist.twist.linear.y
        self.linear = False
        self.angular = False

        self.timestamp = timestamp
        self.twist_linear_x = odom_linear_x
        self.twist_linear_y = odom_linear_y

        self.check_drifting()

    def check_drifting(self):
        if self.throttle == float('inf') or self.steering_angle == float('inf') or self.turning_radius == float('inf'):
            return
        if self.theor_ang_vel == float('inf') or self.twist_angular_z == float('inf'):
            return
        if self.linear_acceleration == float('inf'):
            return

        # print('Checking drifting conditions at timestamp:', self.timestamp)

        # Calculate with latest values
        # print('Calculating linear drift condition...')
        odom_comb = 2 * np.sqrt(self.twist_linear_x**2 + self.twist_linear_y**2)
        throttle = self.throttle

        linear_drift_estimate = abs(odom_comb - throttle)
        # print('Linear drift estimate:', linear_drift_estimate)
        # print(f'{self.timestamp},{linear_drift_estimate}')

        if linear_drift_estimate > 2.0:  # Threshold for linear drift
            # print('Linear drift detected at timestamp:', self.timestamp)
            drifting_msg = Bool()
            drifting_msg.data = True
            self.drifting_publisher.publish(drifting_msg)
            if self.timestamp - self.drifting_timestamp > 2.0:
                self.drifting = True
                self.drifting_timestamp = self.timestamp
                self.linear = True
        else:
            if self.timestamp - self.drifting_timestamp > 2.0:
                self.drifting = False

        angular_val = np.deg2rad(self.twist_angular_z)
        theor_ang_val = self.theor_ang_vel

        # print('Calculating angular drift condition...')
        angular_drift_estimate = abs(angular_val - theor_ang_val)
        # print('Angular drift estimate:', angular_drift_estimate)
        # print(f'{self.timestamp},{angular_drift_estimate}')

        if angular_drift_estimate > 3.0:  # Threshold for angular drift
            # print('Angular drift detected at timestamp:', self.timestamp)
            drifting_msg = Bool()
            drifting_msg.data = True
            self.drifting_publisher.publish(drifting_msg)
            if self.timestamp - self.drifting_timestamp > 2.0:
                self.drifting = True
                self.drifting_timestamp = self.timestamp
                self.angular = True
        else:
            if self.timestamp - self.drifting_timestamp > 2.0:
                self.drifting = False

        print(f'Drifting status at {self.timestamp}: {self.linear | self.angular}')
        if self.linear:
            print(f'Linear Mu: {abs(self.linear_acceleration / self.y_lin_accel)}')
        if self.angular:
            mu = ((2*odom_comb)**2 / self.turning_radius) / self.gravity
            print(f'Angular Mu: {mu}')

def main(args=None):
    rclpy.init(args=args)
    drift_detector = DriftDetector()
    rclpy.spin(drift_detector)
    drift_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
