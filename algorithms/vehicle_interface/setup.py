from setuptools import setup
from glob import glob
import os

package_name = 'vehicle_interface'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jairlab',
    maintainer_email='jairlab@example.com',
    description='Vehicle interface adapter for F1TENTH sim and real car outputs.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vehicle_interface_node = vehicle_interface.vehicle_interface_node:main',
        ],
    },
)
