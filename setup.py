from setuptools import setup, find_packages

setup(
    name="json-memory",
    version="1.4.0",
    description="Hierarchical associative memory for AI agents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Dion Christian",
    author_email="dion.christiann@gmail.com",
    url="https://github.com/dioncx/json-memory",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
