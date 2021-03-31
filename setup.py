# coding=utf-8

from pathlib import Path
from setuptools import find_packages, setup

project_root = Path(__file__).parent
install_requires = (project_root / 'requirements.txt').read_text().splitlines()
exec(open('techmo/_version.py').read())

setup(name='techmo-wavelet',
      version=__version__,
      url='https://github.com/techmo-pl/vamp-wavelet-fft',
      author='Mariusz Ziółko, Michał Kucharski',
      author_email='mariusz.ziolko@techmo.pl',
      description='A module for audio features extraction from Techmo',
      install_requires=install_requires,
      long_description=(project_root / 'README.md').read_text(),
      long_description_content_type='text/markdown',
      python_requires='>=3.5.0',
      classifiers=[
          'Programming Language :: Python :: 3 :: Only',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
      ],
      packages=find_packages())
