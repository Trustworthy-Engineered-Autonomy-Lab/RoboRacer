import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


class DriftDetectorNode(Node):

    def __init__(self):
        super().__init__('drift_publisher')
        self.publisher_ = self.create_publisher(Bool, 'is_drifting', 10)
        
        self.odom_subsciber = self.create_subscription(Odom, 'odom', self.odom_callback)
        self.subscriber = self.create_subscription(Imu, 'sensors/imu',self.imu_callback)
        self.is_drifting = Bool()

    def imu_callback(self):
        msg = Bool()
        
        


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
