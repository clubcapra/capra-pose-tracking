from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'capra_pose_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/pose_detection', glob('pose_detection/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='capra',
    maintainer_email='capra@etsmtl.ca',
    description='Pose tracking for rove',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'person_detection_node = capra_pose_tracking.person_detection_node:main',
        ],
    },
)
