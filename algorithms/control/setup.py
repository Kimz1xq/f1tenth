from setuptools import setup
from glob import glob
import os

package_name = 'control'

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
    maintainer='jeonbotdae',
    maintainer_email='jeonbotdae@example.com',
    description='Pure Pursuit controller with sim/real drive output abstraction.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pure_pursuit_node = control.pure_pursuit_node:main',
            'linear_mpc_node = control.linear_mpc_node:main',
            'unicorn_l1_node = control.unicorn_l1_node:main',
        ],
    },
)
