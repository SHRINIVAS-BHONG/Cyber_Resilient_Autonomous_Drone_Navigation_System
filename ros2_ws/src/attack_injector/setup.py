from setuptools import setup

package_name = 'attack_injector'

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
    description='Attack injector node',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'attack_injector_node = attack_injector.attack_injector_node:main',
        ],
    },
)
