# camera_model

Part of [camodocal](https://github.com/hengli/camodocal).

## Dependencies

- ROS 2 (Foxy or later)
- [Google Ceres](http://ceres-solver.org)
- OpenCV
- Eigen3
- ament_cmake

## Building

```bash
# from your ROS 2 workspace root
git clone https://github.com/dvorak0/camera_model.git src/camera_model
colcon build --packages-select camera_model
source install/setup.bash
```

## Calibration

Use the `calibration_node` (formerly `intrinsic_calib.cc`) to perform camera intrinsic calibration:

```bash
ros2 run camera_model calibration_node \
  --input_image_dir <path_to_images> \
  --board_rows <rows> \
  --board_cols <cols> \
  [--square_size <meters>] \
  [--output_calib_file <filename>]
```

## Undistortion

Include the camera model header and use the following interface:

```cpp
#include <camodocal/camera_models/Camera.h>

// liftProjective: from pixel (u,v) to projective ray
Eigen::Vector3d ray = camera->liftProjective({u, v});

// spaceToPlane (Pi function): from 3D point to pixel
Eigen::Vector2d uv = camera->spaceToPlane(point3D);
```

## License

TODO (e.g., MIT, BSD)
