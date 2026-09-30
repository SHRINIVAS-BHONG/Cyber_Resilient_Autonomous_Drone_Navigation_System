from setuptools import setup

package_name = 'estimator'

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
    description='Dual state estimator (Main and Trusted)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'estimator_node = estimator.estimator_node:main',
        ],
    },
)
