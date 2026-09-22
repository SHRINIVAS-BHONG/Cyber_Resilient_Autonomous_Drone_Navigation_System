from setuptools import find_packages, setup

package_name = 'ml_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Shrinivas',
    maintainer_email='shrinivas@example.com',
    description='Real-Time Machine Learning Cyber-Attack Detection Node for Autonomous Drones',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ml_detector_node = ml_detector.ml_detector_node:main',
        ],
    },
)
