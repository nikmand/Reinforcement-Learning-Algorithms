from distutils.core import setup

with open('README.txt', 'r') as long_desc_file:
    long_description = long_desc_file.read()

setup(
    name='rl-library',
    version='1.0.0',
    maintainer='Nikiforos Mandilaras',
    maintainer_email='nikmand@outlook.com',
    packages=['rl-library'],
    url='',
    license='',
    description='',
    long_description=long_description,
    classifiers=[
        'Programming Language :: Python :: 3 :: Only'
    ]
)
