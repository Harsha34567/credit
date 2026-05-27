from setuptools import find_packages, setup

setup(
    name="credit_scoring_model",
    version="0.1.0",
    description="Industry-grade credit scoring ML pipeline and Streamlit application.",
    author="Credit Scoring Team",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=2.3.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.3.3",
    ],
    entry_points={
        "console_scripts": [
            "credit-score=main:main",
        ],
    },
)
