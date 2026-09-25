from pathlib import Path

from setuptools import setup, find_packages

_HERE = Path(__file__).resolve().parent
_LONG_DESCRIPTION = (_HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="gsfu",
    version="0.2.0",
    license="BSD-2-Clause",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
    ],
    description=("Library and command line utility for indexing, reading, and writing "
                  "Generic Sensor Format (GSF) sonar data files."),
    long_description=_LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/valschmidt/gsfu",
    project_urls={
        "Source": "https://github.com/valschmidt/gsfu",
        "Issue Tracker": "https://github.com/valschmidt/gsfu/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "gsfu.py=GSFU.gsfu:main",
            "kmall2gsf.py=GSFU.kmall2gsf:main",
        ],
    },
    keywords="hydrography multibeam sonar generic sensor format gsf",
    author="Val Schmidt",
    author_email="Val.Schmidt@unh.edu",
)
