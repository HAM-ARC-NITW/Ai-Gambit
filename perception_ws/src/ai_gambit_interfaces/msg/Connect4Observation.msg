# Flatten cells bottom-to-top, left-to-right: index = row * 7 + column.
# 0 empty, 1 red, 2 yellow, 255 unknown. Feeder slots are row * 6 + column.
std_msgs/Header header
uint64 observation_id
string status
string reason
bool board_pose_valid
bool feeder_pose_valid
bool scene_stable
bool scene_occluded
uint8[42] cells
float32[42] confidence
uint8[42] feeder
float32[42] feeder_confidence
