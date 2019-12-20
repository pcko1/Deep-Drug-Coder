from setuptools import setup

setup(
    name="ddc_pub",
    version="0.1",
    description="Neural network to encode/decode molecules.",
    url="https://github.com/pcko1/Deep-Drug-Coder",
    author="Panagiotis-Christos Kotsias",
    author_email="kotsias.pan@gmail.com",
    license="GPLv3",
    packages=["ddc_pub"],
    install_requires=[
        "numpy",
        "h5py",
        "ipywidgets",
        #"tensorflow-gpu==1.12.0",
        "keras==2.2.5",
        "tqdm",
        "scikit-learn",
        "scipy",
        "ipykernel",
        "ipython",
        "matplotlib",
        "pandas",
    ],
    zip_safe=False,
)
