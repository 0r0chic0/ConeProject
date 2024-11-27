from glob import glob
import os
from setuptools import setup

package_name = 'lab7_pkg'

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
    description='f1tenth lab7_pkg',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rrt_node = lab7_pkg.rrt_node:main',
            'waypoint_node = lab7_pkg.waypoint_node:main',
            'occupancy_grid_visualizer = lab7_pkg.occupancy_grid_visualizer:main',
            'waypoint_viz_node = lab7_pkg.waypoint_viz_node:main',
            'pure_pursuit_node = lab7_pkg.pure_pursuit_node:main',
        ],
    },
)
