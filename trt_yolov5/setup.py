from setuptools import setup

package_name = 'trt_yolov5'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='phatvo',
    maintainer_email='hug.phat.vo@gmail.com',
    description='ROS2 package for Yolov5 object detection using TensorRT',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trt_detector = trt_yolov5.trt_detection:main',
        ],
    },
)
