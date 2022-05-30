from setuptools import setup, find_packages

with open('./requirements.txt', 'r') as f:
    packages = f.read().splitlines()

setup(
    name='Kernel Autoencoder',
    packages=find_packages(), 
    install_requires=packages
)