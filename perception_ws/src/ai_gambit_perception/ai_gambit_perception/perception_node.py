import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Pose
import cv2
import numpy as np

class AiGambitPerceptionNode(Node):
    def __init__(self):
        super().__init__('ai_gambit_perception_node')
        
        # Dual-task publishers for the downstream teams
        self.chess_publisher = self.create_publisher(String, '/ai_gambit/board_fen', 10)
        self.block_publisher = self.create_publisher(PoseArray, '/ai_gambit/block_states', 10)
        
        # Main loop running at 10 Hz
        self.timer = self.create_timer(0.1, self.perception_loop)
        
        # Chess board configuration variables (preserving your original syntax)
        self.board_size = (400, 400)
        self.M = None
        self.pts_src = np.array([[50, 50], [350, 50], [350, 350], [50, 350]], dtype=np.float32)
        
        # Track last FEN to optimize publishing on-change if desired
        self.last_fen = None
        
        self.get_logger().info("Ai-Gambit Multi-Task Perception Node initialized.")

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
        """Detects stacked blocks using HSV color thresholding and contours."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Example color bounds for block tracking
        lower_bound = np.array([35, 50, 50])
        upper_bound = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        block_poses = []
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 300:
                x, y, w, h = cv2.boundingRect(cnt)
                block_poses.append((x + w // 2, y + h // 2))
                
        return block_poses

    def perception_loop(self):
        # Placeholder frame input (replace with real camera subscription stream later)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # --- 1. Chess Pipeline Execution ---
        warped = self.warp_board(dummy_frame)
        squares = self.segment_squares(warped)
        current_fen = self.extract_fen(squares)
        
        # Publish FEN string
        chess_msg = String()
        chess_msg.data = current_fen
        self.chess_publisher.publish(chess_msg)
        
        # --- 2. Block Stacking Pipeline Execution ---
        blocks = self.process_blocks(dummy_frame)
        
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = self.get_clock().now().to_msg()
        pose_array_msg.header.frame_id = "camera_optical_frame"
        
        for (cx, cy) in blocks:
            p = Pose()
            p.position.x = float(cx)
            p.position.y = float(cy)
            pose_array_msg.poses.append(p)
            
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