from setuptools import setup, find_packages

setup(
    name="edl-pytorch",
    packages=find_packages(exclude=["examples"]),
    version="0.0.2",
    license="MIT",
    description="Evidential Learning in Pytorch",
    author="Teddy Koker",
    author_email="teddy.koker@gmail.com",
    long_description_content_type="text/markdown",
    url="https://github.com/teddykoker/evidential-learning-pytorch",
    install_requires=["torch"],
    classifiers=["Programming Language :: Python :: 3"],
)
