from setuptools import setup

package_name = 'ai_gambit_perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    package_data={package_name: ['workspace.json','connect4.json']},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'opencv-python', 'numpy'],
    zip_safe=True,
    maintainer='HAM',
    description='Multi-task perception module for Ai-Gambit',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = ai_gambit_perception.perception_node:main',
            'connect4_node = ai_gambit_perception.connect4_node:main',
        ],
    },
)