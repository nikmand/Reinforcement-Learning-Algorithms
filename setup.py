from setuptools import setup, find_packages

with open('README.md', 'r') as long_desc_file:
    long_description = long_desc_file.read()

setup(
    name='rlsuite',
    version='1.0.0',
    maintainer='Nikiforos Mandilaras',
    maintainer_email='nikmand@outlook.com',
    packages=find_packages(),
    url='https://github.com/nikmand/Reinforcement-Learning-Library',
    license='',
    description='Implementation of Reinforcement Learning Algorithms',
    long_description=long_description,
    python_requires='>=3.5',
    install_requires=[
        "torch",
        "torchvision",
        "tb-nightly",
        "gym",
        "matplotlib"],
    classifiers=[
        'Programming Language :: Python :: 3 :: Only'
    ]
)
