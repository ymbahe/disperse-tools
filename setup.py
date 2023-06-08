import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="disperse-tools",
    version="0.1",
    author="Yannick Bahe",
    author_email="yannick.bahe@epfl.ch",
    description="Tools for working with the DisPerSE output",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS independent"
    ],
    python_requires='>=3.6'
)

