from setuptools import setup, find_packages
from os import path, environ

cur_dir = path.abspath(path.dirname(__file__))

with open(path.join(cur_dir, 'requirements.txt'), 'r') as f:
    requirements = f.read().split()



setup(
    name='openPMD-wavefront',
    version = 'v0.0.1',
    packages = ['pmd_wavefront'],
    package_dir={'pmd_wavefront':'pmd_wavefront'},
    url='https://github.com/LUME-SIMEX/openPMD-wavefront',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=requirements,
    include_package_data=True,
    python_requires='>=3.6'
)
