from setuptools import setup, find_packages

setup(
    name="json-memory",
    version="0.1.3",
    description="Hierarchical associative memory for AI agents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Alice",
    author_email="dev@example.com",
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
