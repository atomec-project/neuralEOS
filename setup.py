from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="neuralEOS",
    version="0.0.1",
    description="Machine-learned average-atom pressures",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Tim Callow.",
    author_email="t.callow@hzdr.de",
    url="https://github.com/atomec-project/neuralEOS",
    license=license,
    packages=find_packages(exclude=("tests", "docs", "examples")),
    python_requires=">=3.8",
)
