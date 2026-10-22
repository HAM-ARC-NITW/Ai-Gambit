# Arrays are a1,b1,...h1,a2,...h8. 0=empty, 1=occupied, 255=unknown.
std_msgs/Header header
uint64 observation_id
bool board_pose_valid
bool scene_stable
string status
uint8[64] occupancy
float32[64] confidence
# Approximate visual centroid offsets in board-frame metres, not base pose errors.
float32[64] offset_x
float32[64] offset_y
