# -*- coding: utf-8 -*-
# Copyright 2020 moffragmentor authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model agnostic Python implementation of the epsilon-PAL algorithm"""

from setuptools import find_packages, setup

import versioneer

with open("requirements.txt", "r") as fh:
    REQUIREMENTS = fh.readlines()


with open("README.md", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

setup(
    name="moffragmentor",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Splits MOFs into metal nodes and linkers",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["moffragmentor", "moffragmentor.*"]),
    url="https://github.com/kjappelbaum/pymoffragmentor",
    license="Apache 2.0",
    install_requires=REQUIREMENTS,
    extras_require={
        "testing": ["pytest==6.*", "pytest-cov==2.*"],
        "docs": [
            "sphinx==3.*",
            "sphinx-book-theme==0.*",
            "sphinx-autodoc-typehints==1.*",
            "sphinx-copybutton==0.*",
        ],
        "pre-commit": [
            "pre-commit==2.*",
            "pylint==2.*",
            "isort==5.*",
        ],
        "dev": [
            "versioneer==0.*",
            "black==20.*",
        ],
        "julia": ["julia"],
    },
    author="Kevin Jablonka",
    author_email="kevin.jablonka@epfl.ch",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
