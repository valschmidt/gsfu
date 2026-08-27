from setuptools import setup, find_packages


setup(
    name="gsfu",
    version="0.1.0",
    license="BSD-2-Clause",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
    ],
    description=("Library and command line utility for indexing, reading, and writing "
                  "Generic Sensor Format (GSF) sonar data files."),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    entry_points={
        "console_scripts": ["gsfu.py=GSFU.gsfu:main"],
    },
    keywords="hydrography multibeam sonar generic sensor format gsf",
    author="Val Schmidt",
    author_email="Val.Schmidt@unh.edu",
)
