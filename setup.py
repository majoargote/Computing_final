from setuptools import setup, find_packages

setup(
    name = "wine_predict",
    version = "0.1.0",
    packages=find_packages(),
    description = "Wine quality prediction library",
    install_requires = [
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "joblib>=1.0.0",
        "pytest>=7.0.0",
]
)

