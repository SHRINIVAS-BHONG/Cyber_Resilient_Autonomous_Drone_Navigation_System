from setuptools import setup

package_name = 'px4_controller'

setup(
    name=package_name,
    version='2.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Resilience Team',
    maintainer_email='user@example.com',
    description='PX4 and flight trajectory controller node',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'controller_node = px4_controller.controller_node:main',
        ],
    },
)
