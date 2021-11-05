# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:57:57 2021

@author: bugra
"""

import setuptools

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()


def parse_requirements(filename):
    with open(filename, encoding='utf-8') as fid:
        requires = [line.strip() for line in fid.readlines() if line]
    return requires

requirements = parse_requirements(requirements.txt)

setuptools.setup(
    name = 'image_processing',
    version = '0.0.0',
    author = 'Bugra Özdemir',
    author_email = 'bugraa.ozdemir@gmail.com',
    description = 'Image processing tools.',
    long_description = 'Tools for the processing of 3D microscopy images.',
    long_description_content_type = "text/markdown",
    url = 'https://github.com/bugraoezdemir/image_processing',
    # license = 'MIT',
    packages = setuptools.find_packages(exclude = ['wrappers']),
    install_requires = requirements
    )
