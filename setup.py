from setuptools import setup, find_packages

setup(
    name='tracerpaws',
    version='0.1',
    description='Cell tracking and visualization for TRACER polarimetry and lightning',
    packages=find_packages(),
    author='Eric Bruning',
    author_email='eric.bruning@gmail.com',
    url='https://github.com/deeplycloudy/TRACER-PAWS-NEXRAD-LMA/',
    license='BSD-3-Clause',
    long_description=open('README.md').read(),
    include_package_data=True,
)