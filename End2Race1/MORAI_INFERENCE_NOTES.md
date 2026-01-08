# MORAI Inference Notes (End2Race)

## What exists today
- `End2Race/` currently focuses on offline training for:
  - LiDAR teacher: `teacher/LiDARBEVTeacher` (PointPillars-based, raw pointcloud input)
  - Camera student (skeleton): `student/CameraBEVStudent`
- There is **no ROS2/MORAI runtime node** in this repo yet (unlike `/home/sws/E2R1/E2R/` which has `end2race_driver.py` + `morai_interface.py`).

## Can a `.pth` from End2Race be used in MORAI?
Yes, with conditions:
- The `.pth` files here are (or should be) `state_dict` checkpoints.
- In MORAI/ROS2 runtime you can load them via:
  - `model = ...same architecture/args...`
  - `model.load_state_dict(torch.load(path, map_location=device))`
  - `model.eval()`

## What you must implement to run in MORAI
### 1) ROS2 node wrapper
- Subscribe to sensor topics and build tensors shaped exactly like training:
  - Teacher (LiDAR): needs per-timestep pointcloud tensors matching voxelizer assumptions (e.g. `(N,4)` x,y,z,intensity).
  - Student (Camera): needs image tensors `(3,H,W)` (or multi-cam) + ego signals (speed, steering_input).
- Publish control outputs to topics expected by your MORAI interface.

### 2) Input compatibility check (important)
- `E2R1/E2R/end2race_driver.py` consumes `sensor_msgs/LaserScan` (1D ranges) and feeds a 360-dim vector.
- `End2Race/teacher/LiDARBEVTeacher` consumes **3D point clouds** and uses PointPillars voxelization.
  - If your MORAI setup only provides `LaserScan`, you cannot directly use this teacher without a 3D point source.
  - If MORAI provides `sensor_msgs/PointCloud2` (x,y,z,intensity), then it can be adapted.

### 3) Build/runtime deps
- Teacher relies on PointPillars CUDA/C++ extensions (`End2Race/setup.py`).
  - Your MORAI runtime environment must have these extensions compiled and compatible with its CUDA/PyTorch version.

## Recommended path
- For quickest MORAI deployment:
  - Implement a ROS2 node for the **camera student** first (image input is usually straightforward).
  - Keep teacher inference offline (for cache/KD) unless you have PointCloud2 and the extensions built in the runtime.

