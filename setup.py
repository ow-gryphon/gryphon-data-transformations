import json
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('gryphon_requirements.txt') as fr:
    requirements = fr.read().strip().split('\n')

with open('metadata.json') as fr:
    metadata = json.load(fr)

setuptools.setup(
    name="gryphon-data-transformations",  # Name of the repository
    version="0.0.7",
    author="Daniel Uken",
    author_email="daniel.uken@oliverwyman.com",
    description="Data transformations for general data cleaning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",  # Repository URL or externally maintained page
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=requirements,
)
