from distutils.core import setup

with open('README.md', 'r') as long_desc_file:
    long_description = long_desc_file.read()

setup(
    name='rlsuite',
    version='1.0.0',
    maintainer='Nikiforos Mandilaras',
    maintainer_email='nikmand@outlook.com',
    packages=['rlsuite'],
    url='https://github.com/nikmand/reinforcement-learning-library',
    license='',
    description='Implementation of Reinforcement Learning Algorithms',
    long_description=long_description,
    classifiers=[
        'Programming Language :: Python :: 3 :: Only'
    ]
)
