
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tfmpl==1.0.2',
                     'librosa==0.6.2',
                     'soundfile==0.10.2'
                     ]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='MSYNC trainer application'
)
