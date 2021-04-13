from setuptools import setup, find_packages

setup(
    name="visualize_model_results",
    version="0.1",
    description="Most useful plots to assess ML models performance",
    url="https://github.com/Jordi-Ab/visualize_model_results",
    author="Giordi Azonos Beverido",
    author_email="jordi_a3@hotmail.com",
    packages=find_packages(
        include=[
            "features_pipeline", 
            "features_pipeline.*"
        ]
    ),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "sklearn"
    ],
    zip_safe=False,
)