"""ROS transport for the SAME Connect 4 detector used by the Windows demo.

Observations never alter the official engine state. An action waits for fresh
settled exposures with feedback/cancellation; a short service sets the baseline.
Only this node owns board/feeder TF. The camera source owns camera-to-base TF.
"""
from pathlib import Path
import json
import threading
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer,GoalResponse,CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor,ExternalShutdownException
from rclpy.qos import QoSProfile,ReliabilityPolicy,qos_profile_sensor_data
from rclpy.time import Time
from rclpy.duration import Duration
from cv_bridge import CvBridge,CvBridgeError
from sensor_msgs.msg import Image,CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer,TransformListener,TransformBroadcaster,TransformException
from ai_gambit_interfaces.msg import Connect4Observation
from ai_gambit_interfaces.srv import Connect4Baseline
from ai_gambit_interfaces.action import ObserveConnect4
from .connect4 import Perception,Gate,load_config,usable,overlay,transform


def seconds(stamp):
    return stamp.sec+stamp.nanosec*1e-9


def pose_matrix(message):
    p,q = message.transform.translation,message.transform.rotation
    x,y,z,w = q.x,q.y,q.z,q.w
    rotation = [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    return transform(rotation,[p.x,p.y,p.z])


class Connect4Node(Node):
    def __init__(self):
        super().__init__("connect4_perception")
        self.declare_parameter("config","")
        self.declare_parameter("empty_reference",str(Path.home()/".ros/ai_gambit_connect4/empty_workspace.png"))
        self.declare_parameter("max_frame_age",1.0)
        cfg = load_config(self.get_parameter("config").value or None)
        cfg["max_frame_age"] = self.get_parameter("max_frame_age").value
        self.detector,self.gate = Perception(cfg),Gate(cfg)
        self.reference_path = Path(self.get_parameter("empty_reference").value)
        self.bridge,self.info = CvBridge(),None
        self.latest,self.robot_executing,self.busy = None,False,False
        self.reset_requested = False
        self.condition,self.processing = threading.Condition(),threading.Lock()
        self.last_image = time.monotonic()
        self.last_status = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer,self)
        self.tf_pub = TransformBroadcaster(self)
        group = ReentrantCallbackGroup()
        qos = QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
        self.board_pub = self.create_publisher(Connect4Observation,"/connect4/board_observation",10)
        # Separate engine events from the continuous camera observations. Only
        # successful verify actions publish here; snapshots and resets do not.
        self.move_pub = self.create_publisher(String,"/connect4/verified_move",10)
        self.status_pub = self.create_publisher(String,"/connect4/perception_status",10)
        self.debug_pub = self.create_publisher(Image,"/connect4/debug_image",qos_profile_sensor_data)
        self.create_subscription(CameraInfo,"/connect4/camera/camera_info",self.on_info,qos,callback_group=group)
        self.create_subscription(Image,"/connect4/camera/image_raw",self.on_image,qos,callback_group=group)
        self.create_subscription(String,"/connect4/execution_status",self.on_execution,10,callback_group=group)
        self.create_service(Connect4Baseline,"/connect4/set_baseline",self.baseline,callback_group=group)
        self.action = ActionServer(self,ObserveConnect4,"/connect4/observe",execute_callback=self.observe,
            goal_callback=self.admit,cancel_callback=lambda _:CancelResponse.ACCEPT,callback_group=group)
        self.create_timer(.25,self.watchdog,callback_group=group)

    def on_info(self,msg):
        self.info = msg

    def invalidate(self,status,reason):
        with self.condition:
            self.latest = None
            # Image processing owns the detector. Other callbacks request a
            # reset instead of modifying its counters mid-frame.
            self.reset_requested = True
            self.condition.notify_all()
        if status!=self.last_status:
            self.get_logger().info(status+": "+reason)
            self.last_status = status
        self.status_pub.publish(String(data=status))

    def on_execution(self,msg):
        self.robot_executing = msg.data in ("ROBOT_EXECUTING","PAUSED","RECOVERING")
        if self.robot_executing:
            self.invalidate("ROBOT_EXECUTING","Waiting until execution ends")

    def watchdog(self):
        if time.monotonic()-self.last_image>self.detector.config["max_frame_age"]:
            self.invalidate("CAMERA_FAILURE","Camera stream timed out")

    def message(self,obs):
        msg = Connect4Observation()
        ns = round(obs.timestamp*1e9)
        msg.header.stamp.sec,msg.header.stamp.nanosec = ns//10**9,ns%10**9
        msg.header.frame_id = "connect4_board_frame"
        msg.observation_id,msg.status,msg.reason = obs.observation_id,obs.status,obs.reason
        calibrated = obs.status not in ("CALIBRATION_INVALID","CAMERA_FAILURE","CAMERA_QUALITY")
        msg.board_pose_valid = "board" in obs.poses and calibrated
        msg.feeder_pose_valid = "feeder" in obs.poses and calibrated
        msg.scene_stable,msg.scene_occluded = obs.stable,obs.status=="OCCLUDED"
        msg.cells,msg.confidence = obs.cells,obs.confidence
        msg.feeder,msg.feeder_confidence = obs.feeder,obs.feeder_confidence
        return msg

    def publish(self,obs):
        message = self.message(obs)
        self.board_pub.publish(message)
        self.status_pub.publish(String(data=obs.status))
        debug = self.bridge.cv2_to_imgmsg(overlay(obs),encoding="rgb8")
        debug.header = message.header
        self.debug_pub.publish(debug)
        if obs.status!=self.last_status:
            self.get_logger().info(obs.status+": "+obs.reason)
            self.last_status = obs.status
        # Valid TF is published before waking action/service clients. Exact-time
        # consumers use the identical stamp from the returned observation.
        if not message.board_pose_valid or not message.feeder_pose_valid:
            return
        for group in ("board","feeder"):
            T = np.asarray(obs.poses[group])
            vector = cv2.Rodrigues(T[:3,:3])[0].ravel()
            angle = np.linalg.norm(vector)
            xyz = vector/angle*np.sin(angle/2) if angle>1e-10 else np.zeros(3)
            msg = TransformStamped()
            msg.header.stamp,msg.header.frame_id = message.header.stamp,"base_link"
            msg.child_frame_id = "connect4_"+group+"_frame"
            msg.transform.translation.x,msg.transform.translation.y,msg.transform.translation.z = map(float,T[:3,3])
            msg.transform.rotation.x,msg.transform.rotation.y,msg.transform.rotation.z = map(float,xyz)
            msg.transform.rotation.w = float(np.cos(angle/2))
            self.tf_pub.sendTransform(msg)

    def on_image(self,msg):
        if not self.processing.acquire(blocking=False):
            return  # Drop overlapping processing; never queue stale images.
        try:
            if self.reset_requested:
                self.detector.reset_stability()
                self.reset_requested = False
            info = self.info
            if info is None:
                self.invalidate("CALIBRATION_INVALID","No CameraInfo")
                return
            if (info.width!=msg.width or info.height!=msg.height or info.header.frame_id!=msg.header.frame_id or
                (seconds(info.header.stamp) and abs(seconds(info.header.stamp)-seconds(msg.header.stamp))>.1) or
                info.distortion_model not in ("plumb_bob","rational_polynomial")):
                raise ValueError("Image and CameraInfo calibration/frame/timestamp mismatch")
            self.detector.K,self.detector.D = np.asarray(info.k).reshape(3,3),np.asarray(info.d)
            camera_tf = self.tf_buffer.lookup_transform("base_link",msg.header.frame_id,
                Time.from_msg(msg.header.stamp),timeout=Duration(seconds=.15))
            self.detector.base_camera = pose_matrix(camera_tf)
            if not self.detector.reference:
                if not self.reference_path.is_file():
                    self.invalidate("INITIALIZING","Waiting for the empty reference photograph")
                    return
                image = cv2.imread(str(self.reference_path))
                if image is None:raise ValueError("Unreadable empty reference")
                self.detector.set_empty_reference(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            rgb = self.bridge.imgmsg_to_cv2(msg,desired_encoding="rgb8")
            obs = self.detector.process(rgb,seconds(msg.header.stamp),self.get_clock().now().nanoseconds*1e-9,self.robot_executing)
            self.last_image = time.monotonic()
            if self.reset_requested:
                return
            self.publish(obs)
            with self.condition:
                self.latest = obs
                self.condition.notify_all()
        except (ValueError,cv2.error,CvBridgeError,TransformException) as error:
            self.invalidate("CALIBRATION_INVALID",str(error))
        finally:
            self.processing.release()

    def baseline(self,request,response):
        try:
            with self.condition:
                if self.latest is None or self.robot_executing or self.busy:
                    self.gate.baseline = None
                    raise ValueError("No usable image, robot executing, or observation action busy")
                self.gate.begin(self.latest,list(request.expected),list(request.expected_feeder),
                                request.game_version,request.operation_id)
            response.success,response.reason = True,"Baseline matches confirmed board and supply"
        except ValueError as error:
            response.success,response.reason = False,str(error)
        return response

    def admit(self,request):
        with self.condition:
            if (self.busy or request.mode not in ("snapshot","verify","reset") or not request.operation_id or
                not np.isfinite(request.timeout_seconds) or not 0<request.timeout_seconds<=15):
                return GoalResponse.REJECT
            self.busy = True
        return GoalResponse.ACCEPT

    def observe(self,goal):
        req,result = goal.request,ObserveConnect4.Result()
        start = time.monotonic()
        try:
            with self.condition:
                sequence = self.latest.observation_id if self.latest else -1
            while rclpy.ok() and time.monotonic()-start<req.timeout_seconds:
                if goal.is_cancel_requested:
                    with self.condition:self.gate.baseline = None
                    result.reason = "Canceled; require a new baseline before verification"
                    goal.canceled()
                    return result
                with self.condition:
                    obs = self.latest
                    if not self.robot_executing and obs and usable(obs,self.detector.config,seconds(req.after),sequence):
                        result.observation = self.message(obs)
                        if req.mode=="verify":
                            event = self.gate.verify(obs,list(req.expected),list(req.expected_feeder),req.game_version,
                                                     req.operation_id,seconds(req.after))
                            self.move_pub.publish(String(data=json.dumps(event)))
                        elif req.mode=="reset":
                            self.gate.baseline = None
                            if any(obs.cells):raise ValueError("Reset failed: discs remain in grid")
                        result.success,result.reason = True,"Fresh stable "+req.mode+" succeeded"
                        goal.succeed()
                        return result
                    status = "ROBOT_EXECUTING" if self.robot_executing else obs.status if obs else "WAITING_FOR_CAMERA"
                    self.condition.wait(.2)
                goal.publish_feedback(ObserveConnect4.Feedback(status=status,elapsed_seconds=float(time.monotonic()-start)))
            raise ValueError("Timed out waiting for fresh, clear, calibrated, settled exposures")
        except ValueError as error:
            result.reason = str(error)
            if req.mode=="verify":
                with self.condition:self.gate.baseline = None
            goal.abort()
            return result
        finally:
            with self.condition:self.busy = False


def main(args=None):
    rclpy.init(args=args)
    node = Connect4Node()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:executor.spin()
    except (KeyboardInterrupt,ExternalShutdownException):pass
    finally:
        executor.shutdown(timeout_sec=2)
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
