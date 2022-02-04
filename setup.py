from setuptools import find_packages, setup

with open("requirements.txt", "r") as lines:
    requirements = lines.read().splitlines()

setup(
    name="modelstore",
    version="0.0.73",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    include_package_data=True,
    description="modelstore is a library for versioning, exporting, storing, and loading machine learning models",
    long_description="Please refer to: https://modelstore.readthedocs.io/en/latest/",
    long_description_content_type="text/markdown",
    url="https://github.com/operatorai/modelstore",
    author="Neal Lathia",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
    ],
    license="Please refer to the readme",
    python_requires=">=3.6",
    install_requires=requirements,
)
