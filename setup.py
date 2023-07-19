import io
import os
import subprocess
import sys

import setuptools

# Package meta-data.
NAME = "Semantic Meaningfulness"
DESCRIPTION = "todo"
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = "https://github.com/JHoelli/Semantic-Meaningfulness"
EMAIL = "hoellig@fzi.de"
AUTHOR = "Jacqueline Hoellig, Aniek Markus, Jef de Slegte, Prachi Bagave"
REQUIRES_PYTHON = ">=3.6.0"

# Package requirements.
base_packages = [
    "carla-recourse @ https://github.com/carla-recourse/CARLA/archive/refs/heads/main.zip",
]


here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, "Semantic_Meaningfulness", "__version__.py")) as f:
    exec(f.read(), about)

# Where the magic happens:
setuptools.setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=base_packages,
    extras_require={
    },
    include_package_data=True,
    license="BSD-3",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    ext_modules=[]
)