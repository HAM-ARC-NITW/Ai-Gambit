"""ROS 2 transport for the same RGB detector used by the Windows simulation.

No dummy frames or guessed FEN. An independently photographed empty workspace
is required. Camera callbacks keep running while services wait for fresh frames.
See PERCEPTION.md for build, calibration and topic/service contracts.
"""
import threading
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from ai_gambit_interfaces.msg import BoardObservation, RackObservation
from ai_gambit_interfaces.srv import Snapshot, SetBaseline, VerifyState
from .perception import Perception, load_config


def seconds(stamp):
    return stamp.sec + stamp.nanosec*1e-9


def quaternion(rotation):
    """Matrix to XYZW quaternion via axis-angle, without a scipy dependency."""
    vector = cv2.Rodrigues(np.asarray(rotation,float))[0].ravel()
    angle = np.linalg.norm(vector)
    xyz = vector/angle*np.sin(angle/2) if angle > 1e-10 else np.zeros(3)
    return [*xyz, float(np.cos(angle/2))]


class ChessPerceptionNode(Node):
    def __init__(self):
        super().__init__("chess_perception_node")
        self.declare_parameter("workspace_config", "")
        self.declare_parameter("empty_reference", "")
        self.detector = Perception(load_config(self.get_parameter("workspace_config").value or None))
        reference = self.get_parameter("empty_reference").value
        image = cv2.imread(reference) if reference else None
        if image is None:
            raise ValueError("Set empty_reference to an empty-workspace photograph")
        self.reference = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.bridge, self.camera_info = CvBridge(), None
        self.condition, self.processing_lock = threading.Condition(), threading.Lock()
        self.latest, self.robot_executing = None, False
        self.last_image_time = time.monotonic()
        group = ReentrantCallbackGroup()
        self.board_pub = self.create_publisher(BoardObservation,"/perception/board_observation",10)
        self.rack_pub = self.create_publisher(RackObservation,"/perception/rack_observation",10)
        self.status_pub = self.create_publisher(String,"/perception/status",10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(CameraInfo,"/camera/camera_info",self.on_info,qos_profile_sensor_data,callback_group=group)
        self.create_subscription(Image,"/camera/image_raw",self.on_image,qos_profile_sensor_data,callback_group=group)
        self.create_subscription(String,"/coordinator/state",self.on_state,10,callback_group=group)
        self.create_service(Snapshot,"/perception/snapshot",self.snapshot,callback_group=group)
        self.create_service(SetBaseline,"/perception/set_baseline",self.set_baseline,callback_group=group)
        self.create_service(VerifyState,"/perception/verify",self.verify,callback_group=group)
        self.create_timer(.25,self.watchdog,callback_group=group)

    def on_info(self, msg):
        self.camera_info = msg

    def on_state(self, msg):
        self.robot_executing = msg.data in ("ROBOT_EXECUTING","PAUSED","RECOVERING")
        if self.robot_executing:
            self.invalidate("ROBOT_EXECUTING")

    def invalidate(self, status):
        self.status_pub.publish(String(data=status))
        with self.condition:
            self.latest = None
            self.detector.stable_count, self.detector.previous = 0,None
            self.condition.notify_all()

    def watchdog(self):
        if time.monotonic()-self.last_image_time > self.detector.config["max_frame_age"]:
            self.invalidate("CAMERA_FAILURE")

    def on_image(self, msg):
        # Drop overlapping callbacks instead of accumulating stale camera frames.
        if not self.processing_lock.acquire(blocking=False):
            return
        try:
            info, stamp = self.camera_info, seconds(msg.header.stamp)
            now = self.get_clock().now().nanoseconds*1e-9
            if info is None or (seconds(info.header.stamp) != 0 and abs(seconds(info.header.stamp)-stamp) > .5):
                self.invalidate("CALIBRATION_INVALID")
                return
            if msg.width != info.width or msg.height != info.height or msg.header.frame_id != info.header.frame_id:
                self.invalidate("CALIBRATION_INVALID")
                return
            self.detector.K = np.array(info.k).reshape(3,3)
            self.detector.distortion = np.array(info.d)
            if len(info.d) and info.distortion_model not in ("plumb_bob","rational_polynomial"):
                raise ValueError("Unsupported lens distortion model")
            self.detector.config["image_width"], self.detector.config["image_height"] = info.width, info.height
            if self.detector.reference is None:
                self.detector.set_empty_reference(self.reference)
            rgb = self.bridge.imgmsg_to_cv2(msg,desired_encoding="rgb8")
            obs = self.detector.process(rgb,stamp,now,robot_executing=self.robot_executing)
            self.last_image_time = time.monotonic()
            with self.condition:
                self.latest = obs
                self.condition.notify_all()
            self.publish(obs,msg.header.stamp)
        except (ValueError,cv2.error,CvBridgeError) as error:
            self.get_logger().error(str(error))
            self.invalidate("CALIBRATION_INVALID")
        finally:
            self.processing_lock.release()

    def board_message(self, obs):
        msg = BoardObservation()
        ns = int(obs.timestamp*1e9)
        msg.header.stamp.sec, msg.header.stamp.nanosec = ns//10**9, ns%10**9
        msg.header.frame_id = "board_frame"
        msg.observation_id = obs.observation_id
        msg.board_pose_valid = "board" in obs.poses and obs.status != "CALIBRATION_INVALID"
        msg.scene_stable, msg.status = obs.stable, obs.status
        msg.occupancy, msg.confidence = obs.occupancy, obs.confidence
        msg.offset_x = [float(o[0]) for o in obs.offsets]
        msg.offset_y = [float(o[1]) for o in obs.offsets]
        return msg

    def rack_message(self, obs):
        msg = RackObservation()
        msg.header = self.board_message(obs).header
        msg.valid = obs.status == "READY"
        msg.promotion_occupancy, msg.capture_occupancy = obs.rack, obs.tray
        msg.promotion_confidence, msg.capture_confidence = obs.rack_confidence, obs.tray_confidence
        return msg

    def publish(self, obs, stamp):
        self.board_pub.publish(self.board_message(obs))
        self.rack_pub.publish(self.rack_message(obs))
        self.status_pub.publish(String(data=obs.status))
        if obs.status in ("CALIBRATION_INVALID","CAMERA_FAILURE"):
            return
        # This node is the only publisher of these three authoritative frames.
        for name, child in (("board","board_frame"),("rack","promotion_rack_frame"),("tray","capture_tray_frame")):
            if name not in obs.poses:
                continue
            T = np.array(obs.poses[name])
            msg = TransformStamped()
            msg.header.stamp, msg.header.frame_id, msg.child_frame_id = stamp,"base_link",child
            msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float,T[:3,3])
            q = quaternion(T[:3,:3])
            msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w = q
            self.tf.sendTransform(msg)

    def fresh(self, request):
        deadline = time.monotonic()+min(max(request.timeout_seconds,.1),10.)
        with self.condition:
            sequence = self.latest.observation_id if self.latest else -1
            while time.monotonic() < deadline:
                obs = self.latest
                now = self.get_clock().now().nanoseconds*1e-9
                if (obs and obs.observation_id > sequence and obs.status == "READY" and
                    obs.timestamp > seconds(request.after) and now-obs.timestamp <= self.detector.config["max_frame_age"]):
                    return obs
                self.condition.wait(.05)
        return None

    def snapshot(self, request, response):
        response.operation_id, response.game_version = request.operation_id, request.game_version
        obs = self.fresh(request)
        response.success = obs is not None
        response.reason = "Fresh stable observation" if obs else "Timed out waiting for clear calibrated frames"
        if obs:
            response.board, response.rack = self.board_message(obs), self.rack_message(obs)
        return response

    def set_baseline(self, request, response):
        with self.condition:
            obs = self.latest
            response.success, response.reason = (False,"No image") if obs is None else self.detector.establish_baseline(
                obs,list(request.expected),request.game_version,rack=list(request.expected_rack),tray=list(request.expected_tray),
                now=self.get_clock().now().nanoseconds*1e-9)
            if not response.success:
                self.status_pub.publish(String(data="BOARD_MISMATCH"))
        response.operation_id = request.operation_id
        return response

    def verify(self, request, response):
        response.operation_id, response.game_version = request.operation_id, request.game_version
        obs = self.fresh(request)
        response.success, response.reason = (False,"Snapshot timeout") if obs is None else self.detector.verify(
            obs,list(request.expected),version=request.game_version,after=seconds(request.after),
            rack=list(request.expected_rack),tray=list(request.expected_tray),now=self.get_clock().now().nanoseconds*1e-9)
        if obs:
            response.board, response.rack = self.board_message(obs), self.rack_message(obs)
        if not response.success:
            self.status_pub.publish(String(data="BOARD_MISMATCH"))
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ChessPerceptionNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
