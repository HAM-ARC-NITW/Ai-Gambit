import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import numpy as np

class ChessPerceptionNode(Node):
    def __init__(self):
        super().__init__('chess_perception_node')
        
        # Publisher for downstream chess engine and control teams
        self.fen_publisher = self.create_publisher(String, '/ai_gambit/board_fen', 10)
        
        # Timer running at 10 Hz for frame processing loop
        self.timer = self.create_timer(0.1, self.perception_loop)
        
        self.board_size = (400, 400)
        self.M = None  # Homography matrix
        
        # Dummy calibration points for initialization test (replace with real corners later)
        self.pts_src = np.array([[50, 50], [350, 50], [350, 350], [50, 350]], dtype=np.float32)
        
        self.get_logger().info("Ai-Gambit Perception Node has been initialized.")

    def calibrate_board(self, frame):
        pts_dst = np.array([
            [0, 0],
            [self.board_size[0], 0],
            [self.board_size[0], self.board_size[1]],
            [0, self.board_size[1]]
        ], dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(self.pts_src, pts_dst)

    def warp_board(self, frame):
        if self.M is None:
            self.calibrate_board(frame)
        return cv2.warpPerspective(frame, self.M, self.board_size)

    def segment_squares(self, warped_frame):
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

    def perception_loop(self):
        # Placeholder test frame generation (In production, replace with webcam or simulation topic subscription)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        warped = self.warp_board(dummy_frame)
        squares = self.segment_squares(warped)
        fen_string = self.extract_fen(squares)
        
        # Publish FEN string for other teams
        msg = String()
        msg.data = fen_string
        self.fen_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ChessPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()