from setuptools import setup

package_name = 'residual_detector'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Developer',
    maintainer_email='developer@example.com',
    description='Residual-based cyber-attack detector node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'residual_detector_node = residual_detector.residual_detector_node:main',
        ],
    },
)
