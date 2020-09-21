# coding=utf-8

from pathlib import Path

from setuptools import find_packages, setup

project_root = Path(__file__).parent

install_requires = (project_root / 'requirements.txt').read_text().splitlines()

setup(name='Techmo Extracting Features Project',
      version='0.1',
      python_requires='>=3.5.0',
      author='Mariusz Ziółko, Michal Kucharski',
      author_email="mariusz.ziolko@techmo.pl",
      install_requires=install_requires,
      long_description=(project_root / 'README.md').read_text(),
      long_description_content_type="text/markdown",
      packages=find_packages())
