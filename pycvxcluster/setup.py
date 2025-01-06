from setuptools import setup, find_packages

setup(
    name="pycvxcluster",
    version="0.0.1",
    description="some convex clustering algorithms",
    author="Daniel Li",
    packages=find_packages(include=["pycvxcluster", "pycvxcluster.*"], exclude=["tests"]),
    include_package_data=True,
    install_requires=[
        "scikit-learn",
        "scipy",
        "scikit-sparse",
        "numpy"],
    python_requires=">=3.8",
)