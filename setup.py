from setuptools import setup, find_packages


# The text of the README file
with open('README.md') as f:
    rm = f.read()

# This call to setup() does all the work
setup(
    name="freelunch",
    version="0.0.7",
    description="Heuristic and meta-heuristic optimisation suite in Python",
    long_description=rm,
    long_description_content_type="text/markdown",
    url="https://github.com/MDCHAMP/FreeLunch",
    author="Max Champneys",
    author_email="max.champneys@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=['freelunch'],
    package_dir={'':'src'},
    include_package_data=False,
    install_requires=[
        "numpy",
        "scipy"
    ],
)