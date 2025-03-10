import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from vesc_msgs.msg import VescImuStamped
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String, Float64
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from queue import Queue

class DriftingDetector(Node):
    def __init__(self):
        super().__init__('drifting_detector')
        
        self.wheelbase = 0.32  # 32 cm in meters
        self.tirewidth = 0.04445
        self.mass = 3.333
        self.gravity = 9.81
        self.force_of_gravity = self.mass * self.gravity
        
        self.ackermann_subscriber = self.create_subscription(
            AckermannDriveStamped,
            '/ackermann_cmd',
            self.ackermann_callback,
            10
        )
        
        self.imu_subscriber = self.create_subscription(
            VescImuStamped,
            '/sensors/imu',
            self.imu_callback,
            10
        )
            
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.odom_callback,
            10
        )
        
        self.drifting_publisher = self.create_publisher(Bool, 'is_drifting', 10)
        self.angular_vel_publisher = self.create_publisher(String, 'angular_vel', 10)
        self.linear_vel_publisher = self.create_publisher(String, 'linear_vel', 10)
        self.friction_publisher = self.create_publisher(Float64, 'friction_mu', 10)

        self.current_velocity = 0.0
        self.linear_velocity = 0.0
        self.current_steering_angle = 0.0
        self.current_angular_velocity = 0.0
        self.historical_acc = []
        self.time_stamps = []

    def ackermann_callback(self, msg):
        self.current_velocity = msg.drive.speed
        self.current_steering_angle = msg.drive.steering_angle

    def imu_callback(self, msg):
        self.current_angular_velocity = msg.imu.angular_velocity.z
        self.linear_acceleration = msg.imu.linear_acceleration.y
        
        if len(self.historical_acc) < 10:
            self.historical_acc.append(self.linear_acceleration)
            self.time_stamps.append(msg.header.stamp.sec + (msg.header.stamp.nanosec * 10**-9))
        else:
            self.time_stamps = self.time_stamps[1:]
            self.time_stamps.append(msg.header.stamp.sec + (msg.header.stamp.nanosec * 10**-9))
            self.historical_acc = self.historical_acc[1:]
            self.historical_acc.append(self.linear_acceleration)
        
        self.check_drifting()
        self.calculate_friction()
    
    def odom_callback(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.linear_velocity = np.sqrt(vx**2 + vy**2)

    def check_drifting(self):
        turning_radius = 0.0
        
        if self.current_steering_angle == 0:
            linear_msg.data = self.linear_velocity
            self.linear_vel_publisher.publish(linear_msg)
            
            velocity_mismatch_thresh = 0.3  # this must be changed to correct error bound
            
            if abs(self.linear_velocity - self.current_velocity) > velocity_mismatch_thresh:
                is_drifting = True
            else:
                is_drifting = False
                
            drifting_msg = Bool()
            drifting_msg.data = is_drifting
            self.drifting_publisher.publish(drifting_msg)
            
        else:
            turning_radius = (self.wheelbase / np.tan(self.current_steering_angle)) + 0.5 * self.tirewidth  # change to odom angular z value instead of manual calculation
            
            if self.current_velocity > 0:  # Prevent division by zero
                theoretical_angular_velocity = self.current_velocity * turning_radius
                
                threshold = 0.5  # change threshold value to match correct error bound
                difference = abs(self.current_angular_velocity - theoretical_angular_velocity)
                is_drifting = difference > threshold
                
                drifting_msg = Bool()
                drifting_msg.data = bool(is_drifting)
                self.drifting_publisher.publish(drifting_msg)
    
    def angular_velocity(self):
        # Calculate and publish angular velocity
        return

    def calculate_friction(self):
        turning_radius = 0.0
        
        if self.current_steering_angle == 0:
            turning_radius = 0
            times = self.time_stamps
            linear_acc = self.historical_acc
            # TODO: logic here to calculate coefficient of friction while driving in a straight line
        else:
            turning_radius = (self.wheelbase / np.tan(self.current_steering_angle)) + 0.5 * self.tirewidth
        
        if turning_radius > 0:
            force_of_friction = self.mass * ((self.current_velocity * self.current_velocity) / turning_radius)
            mu = force_of_friction / self.force_of_gravity
            
            friction_val = Float64()
            friction_val.data = mu
            self.friction_publisher.publish(friction_val)

def main(args=None):
    rclpy.init(args=args)
    drifting_detector = DriftingDetector()
    rclpy.spin(drifting_detector)
    drifting_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

