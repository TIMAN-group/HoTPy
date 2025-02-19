from setuptools import setup, find_packages

setup(
    name="HoTPy",
    py_modules=['HoT'],
    version="0.1",
    author="Dean E. Alvarez",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)