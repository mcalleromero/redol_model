from setuptools import setup

with open("VERSION", "r") as f:
    VERSION = f.readline()
setup(
    name='redol',
    version=VERSION,
    packages=['redol',],
    install_requires=[
        'scikit-learn==0.23.0',
        'pymp-pypi==0.4.2',
    ],
)