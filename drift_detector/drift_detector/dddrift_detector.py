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


class DriftingDetector(Node):
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

        self.throttles = []
        self.steering_angles = []
        self.twist_angular_zs = []
        self.linear_accelerations = []
        self.twist_linear_xs = []
        self.twist_linear_ys = []
        self.drifting_bools = []
        self.theor_ang_vels = []
        self.turning_radius = []
        
        self.counter = 0
        self.cutoffTime = 1.5  # Size of the moving window

        self.linear_drift_threshold = 1.25
        self.angular_drift_threshold = 3.0

        self.slip_acceleration = 0.0
        
    def ackermann_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 10**-9
        throttle = msg.drive.speed
        steering_angle = msg.drive.steering_angle
        turning_radius = (np.tan(np.absolute(steering_angle)) / self.wheelbase)
        theor_ang_vel = throttle * turning_radius
        
        self.steering_angles.append((steering_angle, timestamp))
        self.throttles.append((throttle, timestamp))
        self.theor_ang_vels.append((theor_ang_vel, timestamp))
        self.turning_radius.append((turning_radius, timestamp))
        
        
        for _ in range(len(self.steering_angles)):
            if (self.steering_angles[-1][1] - self.steering_angles[0][1]) > self.cutoffTime:
                self.steering_angles.pop(0)
            else:
                break
            
        for _ in range(len(self.throttles)):
            if (self.throttles[-1][1] - self.throttles[0][1]) > self.cutoffTime:
                self.throttles.pop(0)
            else:
                break
            
        for _ in range(len(self.theor_ang_vels)):
            if (self.theor_ang_vels[-1][1] - self.theor_ang_vels[0][1]) > self.cutoffTime:
                self.theor_ang_vels.pop(0)
            else:
                break
            
        for _ in range(len(self.turning_radius)):
            if (self.turning_radius[-1][1] - self.turning_radius[0][1]) > self.cutoffTime:
                self.turning_radius.pop(0)
            else:
                break
            
        self.check_drifting()
                    
    def imu_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 10**-9
        angular_velocity = msg.angular_velocity.z
        linear_accelerationx = msg.linear_acceleration.x
        
        self.twist_angular_zs.append((angular_velocity, timestamp))
        self.linear_accelerations.append((linear_accelerationx, timestamp))
        
        for _ in range(len(self.twist_angular_zs)):
            if (self.twist_angular_zs[-1][1] - self.twist_angular_zs[0][1]) > self.cutoffTime:
                self.twist_angular_zs.pop(0)
                self.linear_accelerations.pop(0)
            else:
                break
            
        self.check_drifting()
                    
    def odom_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 10**-9
        twistX = msg.twist.twist.linear.x
        twistY = msg.twist.twist.linear.y
        
        self.twist_linear_xs.append((twistX, timestamp))
        self.twist_linear_ys.append((twistY, timestamp))
        
        
        for _ in range(len(self.twist_linear_xs)):
            if (self.twist_linear_xs[-1][1] - self.twist_linear_xs[0][1]) > self.cutoffTime:
                self.twist_linear_xs.pop(0)
            else:
                break
        
        for _ in range(len(self.twist_linear_ys)):
            if (self.twist_linear_ys[-1][1] - self.twist_linear_ys[0][1]) > self.cutoffTime:
                self.twist_linear_ys.pop(0)
            else:
                break
            
        self.check_drifting()
            
            
    def check_drifting(self):
        self.counter += 1
        shortestList = min([self.throttles, self.steering_angles, self.twist_angular_zs, self.twist_linear_xs, self.twist_linear_ys], key=len)
        longestList = max([self.throttles, self.steering_angles, self.twist_angular_zs, self.twist_linear_xs, self.twist_linear_ys], key=len)
        # print("throttles")
        if len(shortestList) == 0:
                return
        
        # print("FUCK")
        shortestTimes = np.array([i[1] for i in shortestList])        
        
        scaler = StandardScaler()
        throttleVals = np.array([i[0] for i in self.throttles]) # Ackermann throttle values interpolated
        throttleTimes = np.array([i[1] for i in self.throttles])
        scaler.fit(throttleVals.reshape(-1,1))
        throttle_scaled = scaler.transform(throttleVals.reshape(-1,1)).squeeze(axis=-1)
        # print("SHIT")
        twist_linear_xVals = np.array([i[0] for i in self.twist_linear_xs]) # Odometry Filtered Linear Velocity Values Interpolation
        twist_linear_yVals = np.array([i[0] for i in self.twist_linear_ys])

        odom_comb = np.sqrt(twist_linear_xVals**2 + twist_linear_yVals**2)
        odomTimes = np.array([i[1] for i in self.twist_linear_xs])
        
        scaler.fit(odom_comb.reshape(-1,1))
        odom_scaled = scaler.transform(odom_comb.reshape(-1,1)).squeeze(axis=-1)
        
        angular_vals = np.array([i[0] for i in self.twist_angular_zs]) # Angular Velocity Values Interpolation
        angular_times = np.array([i[1] for i in self.twist_angular_zs])
        scaler.fit(angular_vals.reshape(-1,1))
        angular_scaled = scaler.transform(angular_vals.reshape(-1,1)).squeeze(axis=-1)
        
        theor_ang_vel_vals = np.array([i[0] for i in self.theor_ang_vels])
        theor_ang_vel_times = np.array([i[1] for i in self.theor_ang_vels])
        scaler.fit(theor_ang_vel_vals.reshape(-1,1))
        theor_ang_scaled = scaler.transform(theor_ang_vel_vals.reshape(-1,1)).squeeze(axis=-1)
        
        # print("BITCH")
        throttle_scaled = np.interp(shortestTimes, throttleTimes, throttle_scaled)
        odom_scaled = np.interp(shortestTimes, odomTimes, odom_scaled)
        angular_scaled = np.interp(shortestTimes, angular_times, angular_scaled)
        theor_ang_scaled = np.interp(shortestTimes, theor_ang_vel_times, theor_ang_scaled)

        diff_list = np.absolute(throttle_scaled - odom_scaled)
        diff_list_ang = np.absolute(angular_scaled - theor_ang_scaled)
        
        if abs(self.steering_angles[-1][0]) < 0.1:
            if diff_list[-1] > self.linear_drift_threshold: # check if drifting
                print("DRIFTING LINEAR")
                self.drifting_bools.append((True, shortestTimes[-1]))
                for _ in range(len(self.drifting_bools)):
                    if (self.drifting_bools[-1][1] - self.drifting_bools[0][1]) > self.cutoffTime:
                        self.drifting_bools.pop(0)
                    else:
                        break

                drifting_bools_vals = np.array([i[0] for i in self.drifting_bools])
                if drifting_bools_vals[-1] and np.count_nonzero(drifting_bools_vals) == 1:
                    mu = abs(self.linear_accelerations[-1][0])
                    print("linear accel", self.linear_accelerations[-1][0] * self.gravity)
                    print(mu)
                    self.mus.append((mu, self.linear_accelerations[-1][1]))

            else:
                print('NOT DRIFTING')
                self.slip_acceleration = self.linear_accelerations[-1][0]
                self.drifting_bools.append((False, shortestTimes[-1]))
                for _ in range(len(self.drifting_bools)):
                    if (self.drifting_bools[-1][1] - self.drifting_bools[0][1]) > self.cutoffTime:
                        self.drifting_bools.pop(0)
                    else:
                        break
        
        else:       
            if diff_list_ang[-1] > self.angular_drift_threshold:
                print("DRIFTING ANGULAR")
                self.drifting_bools.append((True, shortestTimes[-1]))
                for _ in range(len(self.drifting_bools)):
                    if (self.drifting_bools[-1][1] - self.drifting_bools[0][1]) > self.cutoffTime:
                        self.drifting_bools.pop(0)
                    else:
                        break
                    
                drifting_bools_vals = np.array([i[0] for i in self.drifting_bools])
                if drifting_bools_vals[-1] and np.count_nonzero(drifting_bools_vals) == 1:
                    mu = ((2*odom_comb[-1])**2 / self.turning_radius[-1][0]) / self.gravity
                    print("mu", mu)
                    print("velocity", odom_comb[-1])
                    print("turning radius", self.turning_radius[-1][0])
                    print("steering angle", self.steering_angles[-1][0])
                    self.mus.append((mu, self.linear_accelerations[-1][1])) # position x y instead of acceleration !!!
            
            else:
                print('NOT DRIFTING')
                self.drifting_bools.append((False, shortestTimes[-1]))
                for _ in range(len(self.drifting_bools)):
                    if (self.drifting_bools[-1][1] - self.drifting_bools[0][1]) > self.cutoffTime:
                        self.drifting_bools.pop(0)
                    else:
                        break
            

        if self.counter == 5000:
            print("PLOTTING")
            #plt.plot(shortestTimes, throttle_scaled)
            plt.plot(shortestTimes, abs(throttle_scaled - odom_scaled))
            plt.savefig("/home/f1tenth/f1tenth_ws/src/drift_detector/test1.pdf")
            
        
def main(args=None):
    rclpy.init(args=args)
    drifting_detector = DriftingDetector()
    rclpy.spin(drifting_detector)
    drifting_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
        
        
