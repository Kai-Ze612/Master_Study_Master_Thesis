from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'rl_remote_controller'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kai',
    maintainer_email='ge62meq@mytum.de',
    description='RL-based adaptive PD control for teleoperation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'local = rl_remote_controller.local:main',
            'rl_remote = rl_remote_controller.rl_remote:main',
            'rl_training = rl_remote_controller.rl_training:main',
        ],
    },
    # Add this to force ROS2-compatible installation
    script_name='setup.py',
    cmdclass={},
)