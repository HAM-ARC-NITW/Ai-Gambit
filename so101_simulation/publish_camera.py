"""Optional ROS 2 camera source. Use inside Ubuntu/ROS Humble, not native Windows.

Publishes RGB Image and matching CameraInfo with identical timestamps, plus
static base/camera transforms. Run perception_node in a second ROS terminal.
"""
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from perception_sim import Simulation, camera_calibration


class SimulatedCamera(Node):
    def __init__(self):
        super().__init__("mujoco_overhead_camera")
        self.sim, self.bridge = Simulation(), CvBridge()
        self.image_pub = self.create_publisher(Image,"/camera/image_raw",qos_profile_sensor_data)
        self.info_pub = self.create_publisher(CameraInfo,"/camera/camera_info",qos_profile_sensor_data)
        self.tf = StaticTransformBroadcaster(self)
        transforms = []
        for parent,child,pos,rotation in (("base_link","camera_link",self.sim.config["camera_position"],[0.,0.,0.,1.]),
                                          ("camera_link","camera_optical_frame",[0.,0.,0.],[1.,0.,0.,0.])):
            msg = TransformStamped()
            msg.header.frame_id, msg.child_frame_id = parent,child
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.transform.translation.x,msg.transform.translation.y,msg.transform.translation.z = map(float,pos)
            msg.transform.rotation.x,msg.transform.rotation.y,msg.transform.rotation.z,msg.transform.rotation.w = rotation
            transforms.append(msg)
        self.tf.sendTransform(transforms)
        self.create_timer(.2,self.publish)

    def publish(self):
        stamp = self.get_clock().now().to_msg()
        image = self.bridge.cv2_to_imgmsg(self.sim.frame(),encoding="rgb8")
        image.header.stamp, image.header.frame_id = stamp,"camera_optical_frame"
        K,D,_ = camera_calibration(self.sim.config)
        info = CameraInfo()
        info.header = image.header
        info.width,info.height = self.sim.config["image_width"],self.sim.config["image_height"]
        info.k, info.d = K.ravel().tolist(),D.tolist()
        info.r = [1.,0.,0.,0.,1.,0.,0.,0.,1.]
        info.p = [K[0,0],0.,K[0,2],0.,0.,K[1,1],K[1,2],0.,0.,0.,1.,0.]
        info.distortion_model = "plumb_bob"
        self.info_pub.publish(info)
        self.image_pub.publish(image)


def main():
    rclpy.init()
    node = SimulatedCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.sim.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
