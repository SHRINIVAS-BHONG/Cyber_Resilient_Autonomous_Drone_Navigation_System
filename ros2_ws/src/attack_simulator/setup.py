from setuptools import setup

package_name = 'attack_simulator'

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
    description='Cyber-attack simulation and sensor corruption injector node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'attack_injector_node = attack_simulator.attack_injector_node:main',
        ],
    },
)
