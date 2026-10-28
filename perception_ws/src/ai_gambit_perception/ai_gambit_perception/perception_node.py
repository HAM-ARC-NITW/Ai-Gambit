import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Pose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class AiGambitPerceptionNode(Node):
    def __init__(self):
        super().__init__('ai_gambit_perception_node')
        
        # Dual-task publishers for the downstream teams
        self.chess_publisher = self.create_publisher(String, '/ai_gambit/board_fen', 10)
        self.block_publisher = self.create_publisher(PoseArray, '/ai_gambit/block_states', 10)
        self.image_subscriber = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()
        self.latest_frame = None
        
        # Main loop running at 10 Hz
        self.timer = self.create_timer(0.1, self.perception_loop)
        
        # Chess board configuration variables (preserving your original syntax)
        self.board_size = (400, 400)
        self.M = None
        self.pts_src = np.array([[50, 50], [350, 50], [350, 350], [50, 350]], dtype=np.float32)
        
        # Track last FEN to optimize publishing on-change if desired
        self.last_fen = None
        
        self.get_logger().info("Ai-Gambit Multi-Task Perception Node initialized.")

    def image_callback(self, msg):
        """Store the latest camera frame for the timed perception loop."""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().error(f"Could not convert camera image: {exc}")

    def calibrate_board(self, frame):
        """Computes the homography matrix from source points to top-down grid."""
        pts_dst = np.array([
            [0, 0],
            [self.board_size[0], 0],
            [self.board_size[0], self.board_size[1]],
            [0, self.board_size[1]]
        ], dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(self.pts_src, pts_dst)

    def warp_board(self, frame):
        """Warps the raw camera frame into a clean top-down perspective view."""
        if self.M is None:
            self.calibrate_board(frame)
        return cv2.warpPerspective(frame, self.M, self.board_size)

    def segment_squares(self, warped_frame):
        """Slices the warped board into 64 individual square images (a8 to h1)."""
        squares = {}
        sq_w = self.board_size[0] // 8
        sq_h = self.board_size[1] // 8
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
        
        for r_idx, rank in enumerate(ranks):
            for f_idx, file in enumerate(files):
                x1, y1 = f_idx * sq_w, r_idx * sq_h
                x2, y2 = (f_idx + 1) * sq_w, (r_idx + 1) * sq_h
                squares[f"{file}{rank}"] = warped_frame[y1:y2, x1:x2]
        return squares

    def extract_fen(self, squares):
        """Analyzes the 64 squares and constructs the live FEN string."""
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
        fen_rows = []
        
        for rank in ranks:
            empty_count = 0
            row_fen = ""
            for file in files:
                square_img = squares[f"{file}{rank}"]
                gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                variance = np.var(blurred)
                
                # Heuristic threshold for piece occupancy
                if variance < 100.0:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_fen += str(empty_count)
                        empty_count = 0
                    mean_val = np.mean(blurred)
                    row_fen += 'P' if mean_val > 120 else 'p'
            
            if empty_count > 0:
                row_fen += str(empty_count)
            fen_rows.append(row_fen)
            
        return f"{'/'.join(fen_rows)} w KQkq - 0 1"

    def process_blocks(self, frame):
        """Detects colored block tops and returns their image-space centres."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Blocks are saturated; the arm, floor and white target are not.
        mask = cv2.inRange(hsv, np.array([0, 80, 40]), np.array([179, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        block_poses = []

        for cnt in contours:
            if cv2.contourArea(cnt) > 20:
                x, y, w, h = cv2.boundingRect(cnt)
                block_poses.append({
                    'x': x + w / 2.0,
                    'y': y + h / 2.0,
                    'yaw': 0.0,
                    'level': 1,
                })
                
        return block_poses

    def blocks_message(self, frame):
        """Create the ROS message for a BGR frame.

        x and y are pixel coordinates in the supplied image. A consumer that
        knows the camera calibration can transform them into world coordinates.
        """
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = self.get_clock().now().to_msg()
        pose_array_msg.header.frame_id = "camera_optical_frame"
        for block in self.process_blocks(frame):
            p = Pose()
            p.position.x = block['x']
            p.position.y = block['y']
            p.position.z = float(block['level'])  # ROS Point fields require float, including whole levels.
            p.orientation.z = np.sin(block['yaw'] / 2.0)
            p.orientation.w = np.cos(block['yaw'] / 2.0)
            pose_array_msg.poses.append(p)
        return pose_array_msg

    def process_frame(self, frame):
        """Run both pipelines synchronously; used by simulation adapters."""
        warped = self.warp_board(frame)
        squares = self.segment_squares(warped)
        current_fen = self.extract_fen(squares)
        return current_fen, self.blocks_message(frame)

    def perception_loop(self):
        if self.latest_frame is None:
            return
        frame = self.latest_frame

        # --- 1. Chess Pipeline Execution ---
        current_fen, pose_array_msg = self.process_frame(frame)
        
        # Publish FEN string
        chess_msg = String()
        chess_msg.data = current_fen
        self.chess_publisher.publish(chess_msg)
        
        # --- 2. Block Stacking Pipeline Execution ---
        self.block_publisher.publish(pose_array_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AiGambitPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
