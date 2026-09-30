from setuptools import setup

package_name = 'telemetry_logger'

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
    description='Telemetry logger node for recording runs',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'logger_node = telemetry_logger.logger_node:main',
        ],
    },
)
