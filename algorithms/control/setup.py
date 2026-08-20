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
            glob('config/*.yaml') + glob('config/*.csv')
        ),
        (
            os.path.join('share', package_name),
            ['THIRD_PARTY_NOTICES.md']
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jeonbotdae',
    maintainer_email='jeonbotdae@example.com',
    description='Selectable F1TENTH controllers with common sim/real interfaces.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pure_pursuit_node = control.pure_pursuit_node:main',
            'linear_mpc_node = control.linear_mpc_node:main',
            'nonlinear_mpcc_node = control.nonlinear_mpcc_node:main',
            'unicorn_l1_node = control.unicorn_l1_node:main',
            'forza_map_node = control.forza_map_node:main',
        ],
    },
)
