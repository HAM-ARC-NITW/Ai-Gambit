"""Actual ROS integration checks. Start run_connect4_ros.sh run in another terminal.
This checks transport/services/actions/TF; the native suite checks RGB edge cases.
"""
import json
from pathlib import Path
import threading
import time
import uuid
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.duration import Duration
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_ros import Buffer,TransformListener
from action_msgs.msg import GoalStatus
from ai_gambit_interfaces.action import ObserveConnect4
from ai_gambit_interfaces.srv import Connect4Baseline


def wait(future,timeout=20):
    event=threading.Event();future.add_done_callback(lambda _:event.set())
    if not event.wait(timeout):raise TimeoutError("ROS response timed out")
    return future.result()


class Check(Node):
    def __init__(self):
        super().__init__("connect4_integration_check")
        group=ReentrantCallbackGroup()
        self.action=ActionClient(self,ObserveConnect4,"/connect4/observe",callback_group=group)
        self.baseline=self.create_client(Connect4Baseline,"/connect4/set_baseline",callback_group=group)
        self.commands=self.create_publisher(String,"/connect4/simulation/command",10)
        self.execution=self.create_publisher(String,"/connect4/execution_status",10)
        self.receipts,self.images,self.status={}, {},""
        self.moves=[]
        self.create_subscription(String,"/connect4/verified_move",
            lambda m:self.moves.append(json.loads(m.data)),10,callback_group=group)
        self.tf=Buffer();self.listener=TransformListener(self.tf,self)
        self.create_subscription(String,"/connect4/simulation/status",self.receipt,10,callback_group=group)
        self.create_subscription(String,"/connect4/perception_status",lambda m:setattr(self,"status",m.data),10,callback_group=group)
        for topic,name in (("/connect4/camera/image_raw","camera"),("/connect4/debug_image","perception")):
            self.create_subscription(Image,topic,lambda m,n=name:self.image(m,n),qos_profile_sensor_data,callback_group=group)
        self.folder=Path(__file__).resolve().parent/"perception_results/ros"
        self.folder.mkdir(parents=True,exist_ok=True)
        self.results={}

    def receipt(self,msg):
        data=json.loads(msg.data);self.receipts[data["operation_id"]]=data

    def image(self,msg,name):
        rgb=np.frombuffer(msg.data,np.uint8).reshape(msg.height,msg.step)[:,:msg.width*3].reshape(msg.height,msg.width,3)
        self.images[name]=rgb.copy()

    def command(self,command,**values):
        key=str(uuid.uuid4());values.update(command=command,operation_id=key)
        self.commands.publish(String(data=json.dumps(values)))
        deadline=time.monotonic()+8
        while key not in self.receipts and time.monotonic()<deadline:time.sleep(.05)
        assert key in self.receipts,"Simulator command was not acknowledged"
        return Time(seconds=self.receipts[key]["applied_at"]).to_msg()

    def goal(self,mode="snapshot",operation="snapshot",version=0,after=None,cells=None,stock=None,timeout=10.):
        goal=ObserveConnect4.Goal(mode=mode,operation_id=operation,game_version=version,timeout_seconds=timeout)
        goal.after=after or self.get_clock().now().to_msg()
        if cells is not None:goal.expected=cells
        if stock is not None:goal.expected_feeder=stock
        return goal

    def observe(self,**kwargs):
        handle=wait(self.action.send_goal_async(self.goal(**kwargs)))
        assert handle.accepted,"Valid observation goal was rejected"
        return wait(handle.get_result_async(),20).result

    def begin(self,operation,cells,stock,version=0):
        return wait(self.baseline.call_async(Connect4Baseline.Request(operation_id=operation,
            game_version=version,expected=cells,expected_feeder=stock)))

    def save(self,name):
        time.sleep(.2)
        for suffix,image in self.images.items():cv2.imwrite(str(self.folder/f"{name}_{suffix}.png"),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))

    def run(self):
        assert self.action.wait_for_server(timeout_sec=30)
        assert self.baseline.wait_for_service(timeout_sec=10)
        deadline=time.monotonic()+10
        while not self.commands.get_subscription_count() and time.monotonic()<deadline:time.sleep(.1)
        stock=[1]*21+[2]*21;empty=[0]*42
        opening=self.observe();assert opening.success,opening.reason
        assert list(opening.observation.cells)==empty and list(opening.observation.feeder)==stock
        pose=self.tf.lookup_transform("base_link","connect4_board_frame",Time.from_msg(opening.observation.header.stamp),timeout=Duration(seconds=2.))
        p=pose.transform.translation
        error=float(np.linalg.norm(np.array([p.x,p.y,p.z])-[.25886,-.075949,.0490674]))
        assert error<.003,"TF board pose differs from the independent fixture dimensions"
        self.results["tf_origin_error_mm"]=error*1000
        self.save("opening")
        assert self.begin("bad-version",empty,stock).success
        wrong=self.observe(mode="verify",operation="bad-version",version=99,cells=empty,stock=stock)
        assert not wrong.success,"Wrong-version verification was accepted"
        self.results["wrong_version_rejected"]=True
        delayed=self.goal(after=(self.get_clock().now()+Duration(seconds=100)).to_msg())
        handle=wait(self.action.send_goal_async(delayed));assert handle.accepted
        time.sleep(.25);wait(handle.cancel_goal_async());result=wait(handle.get_result_async())
        assert result.status==GoalStatus.STATUS_CANCELED
        self.results["action_cancellation"]=True

        assert self.begin("drop-red",empty,stock).success
        completed=self.command("drop",slot=0,column=3)
        expected=empty.copy();expected[3]=1;stock[0]=0
        red=self.observe(mode="verify",operation="drop-red",cells=expected,stock=stock,after=completed)
        assert red.success,red.reason
        assert list(red.observation.cells)==expected
        self.save("verified_drop")
        assert len(self.moves)==1 and self.moves[0]["row"]==0 and self.moves[0]["column"]==3
        assert self.moves[0]["colour"]==1 and self.moves[0]["game_version"]==1
        assert self.moves[0]["operation_id"]=="drop-red"
        self.results["engine_coordinate_event"]=self.moves[0]
        self.results["verified_red_column_3"]=True
        replay=self.observe(mode="verify",operation="drop-red",cells=expected,stock=stock)
        assert not replay.success
        self.results["replay_rejected"]=True

        assert self.begin("wrong-drop",expected,stock,1).success
        completed=self.command("drop",slot=21,column=2)
        intended=expected.copy();intended[10]=2;stock[21]=0
        bad=self.observe(mode="verify",operation="wrong-drop",version=1,cells=intended,stock=stock,after=completed)
        assert not bad.success,"Wrong-column physical drop was accepted"
        self.save("wrong_column")
        self.results["wrong_column_rejected"]=True
        assert len(self.moves)==1,"Rejected/replayed verification published an engine move"

        self.execution.publish(String(data="ROBOT_EXECUTING"));time.sleep(.3)
        blocked=self.observe(timeout=1.)
        assert not blocked.success
        assert not self.begin("busy",expected,stock,1).success
        self.execution.publish(String(data="WAITING_FOR_HUMAN"))
        self.results["execution_inhibition"]=True
        completed=self.command("reset")
        reset=self.observe(mode="reset",after=completed)
        assert reset.success and not any(reset.observation.cells),reset.reason
        self.results["fresh_reset_verified"]=True

        self.command("camera_off");time.sleep(2.5)
        assert self.status=="CAMERA_FAILURE",self.status
        assert not self.observe(timeout=1.).success
        self.command("camera_on")
        recovered=self.observe();assert recovered.success,recovered.reason
        self.results["camera_watchdog_and_recovery"]=True
        self.save("recovered")
        self.results["passed"]=True
        print(json.dumps(self.results,indent=2),flush=True)


def main():
    rclpy.init();node=Check();executor=MultiThreadedExecutor(num_threads=3)
    executor.add_node(node);threading.Thread(target=executor.spin,daemon=True).start()
    try:node.run()
    except Exception as error:
        node.results.update(passed=False,error=str(error));raise
    finally:
        (node.folder/"result.json").write_text(json.dumps(node.results,indent=2))
        executor.shutdown(timeout_sec=2);node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=="__main__":main()
