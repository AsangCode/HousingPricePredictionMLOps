from setuptools import find_packages, setup
from typing import List


setup(
    name="HousingPricePrediction",
    version="0.0.1",
    author="Asang Kumar Singh",
    author_email="asangkumar6666@gmail.com",
    install_requires=["pytest", "pandas", "numpy", "ensure","scikit-learn", "tox"],
    packages=find_packages()
)
