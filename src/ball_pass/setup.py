from setuptools import find_packages, setup

package_name = 'ball_pass'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eunbyeol',
    maintainer_email='rhdmsquf17@inha.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pass_goal = ball_pass.pass_goal_node:main',
            'shoot_goal = ball_pass.shoot_goal_node:main',
            'kick = ball_pass.kick_node:main',
            'rotate = ball_pass.rotate_node:main',
        ],
    },
)
