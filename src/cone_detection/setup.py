from glob import glob
import os
from setuptools import setup

package_name = 'cone_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sathish gopalakrishnan',
    maintainer_email='sathish@ece.ubc.ca',
    description='f1tenth cone_detection',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pure_pursuit_node = cone_detection.pure_pursuit_node:main',
            'camera_node = cone_detection.camera_node:main',
        ],
    },
)
