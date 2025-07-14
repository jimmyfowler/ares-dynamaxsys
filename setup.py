from setuptools import setup

setup(
    name="dynamaxsys",
    version="0.0.6",
    description="dynamical systems with jax",
    author="Karen Leung",
    author_email="kymleung@uw.edu",
    packages=["dynamaxsys"],
    install_requires=[
        "jax",
        "numpy",
    ],
)
