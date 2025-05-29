from setuptools import setup
import os
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
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='ROS2 Franka MuJoCo Controller',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'mujoco_controller = franka_mujoco_controller.mujoco_controller:main',
        'advanced_pd_controller = franka_mujoco_controller.advanced_pd_controller:main',
        'object_pushing_controller = franka_mujoco_controller.object_pushing_controller:main',
        'test_pushing = franka_mujoco_controller.test_pushing:main',
    ],
},
)
