from setuptools import find_packages, setup

package_name = 'ball_detection'

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
    maintainer_email='eunbyeol@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'yolo_detection = ball_detection.yolo_detector:main',
        'yolo_plus_cv = ball_detection.yolo_plus_cv:main',
        ],
    },
)
