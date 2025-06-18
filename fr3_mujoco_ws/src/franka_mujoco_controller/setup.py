from setuptools import setup
import os

# This is used to find files 
from glob import glob

package_name = 'franka_mujoco_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'models'), 
            glob('models/*.xml')),
        (os.path.join('share', package_name, 'models/meshes'), 
            glob('models/meshes/*.stl')),
        (os.path.join('share', package_name, 'config'), 
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kai-Ze',
    maintainer_email='ge62meq@mytum.de',
    description='ROS2 Franka MuJoCo Controller',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'Bridge_ROS2_MUJOCO = franka_mujoco_controller.Bridge_ROS2_MUJOCO:main',
        'position_control = franka_mujoco_controller.position_control:main',
        'teleoperation = franka_mujoco_controller.teleoperation:main',
        'teleoperation_2_local = franka_mujoco_controller.teleoperation_2_local:main',
        'teleoperation_2_remote = franka_mujoco_controller.teleoperation_2_remote:main',
    ],
},
)
