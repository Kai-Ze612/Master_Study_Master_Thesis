# ROS2 bag
ROS2 bag is a data recording and playback system 

## Record all topics
```bash
ros2 bag record -a
```

## Record specific topics
```bash
ros2 bag record /joint_states /cmd_vel /camera/image
```

## Record with custom name
```bash
ros2 bag record /topic1 /topic2 -o my_experiment_data
```

## Data playback
```bash
ros2 bag play my_experiment_data
```
### Play at different speeds
```bash
ros2 bag play my_experiment_data --rate 2.0  # 2x speed
```
```bash
ros2 bag play my_experiment_data --rate 0.5  # Half speed
```