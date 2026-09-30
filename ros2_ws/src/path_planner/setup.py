from setuptools import setup

package_name = 'path_planner'

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
    description='A* path planner node with dynamic obstacle inflation',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'planner_node = path_planner.planner_node:main',
        ],
    },
)
