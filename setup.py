# from distutils.core import setup
from setuptools import setup

with open('README.md', 'r') as long_desc_file:
    long_description = long_desc_file.read()

setup(
    name='rlsuite',
    version='1.0.0',
    maintainer='Nikiforos Mandilaras',
    maintainer_email='nikmand@outlook.com',
    packages=['rlsuite', 'rlsuite.agents', 'rlsuite.nn'],
    url='https://github.com/nikmand/reinforcement-learning-library',
    license='',
    description='Implementation of Reinforcement Learning Algorithms',
    long_description=long_description,
    python_requires='>=3.5',
    install_requires=[
        "-f https://download.pytorch.org/whl/torch_stable.html"
        "torch==1.5.0+cpu",
        "torchvision==0.6.0+cpu",
        "tb-nightly",
        "gym",
        "matplotlib"],
    classifiers=[
        'Programming Language :: Python :: 3 :: Only'
    ]
)
