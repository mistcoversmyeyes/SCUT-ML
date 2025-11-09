
from setuptools import find_packages
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='pzmllog',
      version='1.0.0',
      description='logger designed for machine learning',
      author='unitclock',
      author_email='yang.open@outlook.com',
      requires= ["PyYAML","pynvml","psutil"], 
      packages=find_packages(),  
      license="The MIT License (MIT)",
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
      )
