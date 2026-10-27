"""ROS camera source for the perception demo; the detector stays in its own node.

Run inside WSL with the ROS environment sourced. Commands below only arrange the
test environment; there is no physical arm/MoveIt control in this camera source.
"""
import json
from pathlib import Path
import queue
import time
import cv2
import numpy as np
import mujoco
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from perception_sim import Simulation
from ai_gambit_perception.connect4 import calibration


def main(args=None):
    rclpy.init(args=args)
    node = Node("connect4_simulated_camera")
    node.declare_parameter("asset_dir",str(Path.home()/".ros/ai_gambit_connect4"))
    sim = Simulation(Path(node.get_parameter("asset_dir").value))
    image_pub = node.create_publisher(Image,"/connect4/camera/image_raw",qos_profile_sensor_data)
    info_pub = node.create_publisher(CameraInfo,"/connect4/camera/camera_info",qos_profile_sensor_data)
    commands = queue.Queue()
    status_pub = node.create_publisher(String,"/connect4/simulation/status",10)
    node.create_subscription(String,"/connect4/simulation/command",lambda msg:commands.put(msg.data),10)
    broadcaster = StaticTransformBroadcaster(node)
    K,D,T = calibration(sim.config)
    vector = cv2.Rodrigues(T[:3,:3])[0].ravel()
    angle = np.linalg.norm(vector)
    xyz = vector/angle*np.sin(angle/2)
    tf = TransformStamped()
    tf.header.frame_id,tf.child_frame_id = "base_link","connect4_camera_optical_frame"
    tf.header.stamp = node.get_clock().now().to_msg()
    tf.transform.translation.x,tf.transform.translation.y,tf.transform.translation.z = map(float,T[:3,3])
    tf.transform.rotation.x,tf.transform.rotation.y,tf.transform.rotation.z = map(float,xyz)
    tf.transform.rotation.w = float(np.cos(angle/2))
    broadcaster.sendTransform(tf)
    start,last_camera,enabled = time.monotonic(),0.,True
    initial_time = sim.data.time
    try:
        while rclpy.ok():
            rclpy.spin_once(node,timeout_sec=0.)
            while not commands.empty():
                try:
                    cmd = json.loads(commands.get_nowait())
                    if cmd["command"]=="drop":sim.release(int(cmd["slot"]),int(cmd["column"]))
                    elif cmd["command"]=="reset":sim.load(np.zeros((6,7),np.uint8))
                    elif cmd["command"]=="camera_off":enabled = False
                    elif cmd["command"]=="camera_on":enabled = True
                    else:raise ValueError("Unknown simulation command")
                    # Acknowledges command application, not a settled disc or a
                    # completed robot grasp. Perception must still observe it.
                    status_pub.publish(String(data=json.dumps(dict(operation_id=cmd.get("operation_id",""),
                        success=True,applied_at=node.get_clock().now().nanoseconds*1e-9))))
                except (ValueError,KeyError,TypeError) as error:
                    node.get_logger().error("Simulation command refused: "+str(error))
            # Preserve all physics steps; cap catch-up work so ROS keeps spinning.
            target = initial_time+time.monotonic()-start
            for _ in range(500):
                if sim.data.time>=target:break
                mujoco.mj_step(sim.model,sim.data)
            now = time.monotonic()
            if enabled and now-last_camera>.25:
                captured = node.get_clock().now().to_msg()
                rgb,_ = sim.frame()
                image = Image()
                image.header.stamp,image.header.frame_id = captured,tf.child_frame_id
                image.height,image.width = rgb.shape[:2]
                image.encoding,image.step,image.data = "rgb8",image.width*3,rgb.tobytes()
                info = CameraInfo()
                info.header,info.width,info.height = image.header,image.width,image.height
                info.k,info.d,info.distortion_model = K.ravel().tolist(),D.tolist(),"plumb_bob"
                info.r = [1.,0.,0.,0.,1.,0.,0.,0.,1.]
                info.p = [K[0,0],0.,K[0,2],0.,0.,K[1,1],K[1,2],0.,0.,0.,1.,0.]
                # SIGTERM may shut ROS down during a render. Do not publish
                # into the invalidated context while the runner is closing.
                if not rclpy.ok():break
                info_pub.publish(info);image_pub.publish(image)
                last_camera = now
            time.sleep(.001)
    except KeyboardInterrupt:pass
    finally:
        sim.close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=="__main__":main()
