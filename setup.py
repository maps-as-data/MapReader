#!/usr/bin/env python

from setuptools import setup, find_packages
import sys
import piffle

LONG_DESCRIPTION = None
try:
    # read the description if it's there
    with open('README.md') as desc_f:
        LONG_DESCRIPTION = desc_f.read()
except:
    pass

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: Apache Software License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Topic :: Software Development :: Libraries :: Python Modules',
]

test_requirements = ['pytest>=3.6', 'pytest-cov']

setup(
    name='piffle',
    version=piffle.__version__,
    author='The Center for Digital Humanities at Princeton',
    author_email='cdhdevteam@princeton.edu',
    url='https://github.com/princeton-cdh/piffle',
    license='Apache License, Version 2.0',
    packages=find_packages(),
    install_requires=['requests', 'cached-property', 'attrdict'],
    setup_requires=['pytest-runner'],
    tests_require=test_requirements,
    extras_require={
        'test': test_requirements
    },
    description='Python library for generating IIIF Image API urls',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    classifiers=CLASSIFIERS,
)
