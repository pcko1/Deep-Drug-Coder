from setuptools import setup

setup(
    name="ddc_pub",
    version="0.3",
    description="Neural network to encode/decode molecules.",
    url="https://github.com/pcko1/Deep-Drug-Coder",
    author="Panagiotis-Christos Kotsias",
    author_email="kotsias.pan@gmail.com",
    license="MIT",
    packages=["ddc_pub"],
    install_requires=[
        "numpy          == 1.16.5",
        "h5py           == 2.9.0",
        "tensorflow-gpu == 2.0.0",
        "tqdm           == 4.35.0",
        "scikit-learn   == 0.21.3",
        "scipy          == 1.3.1",
        "ipykernel      == 5.1.1",
        "ipython",
        "matplotlib     == 3.1.1",
        "pandas         == 0.25.1",
	"molsets        == 0.2"
    ],
    zip_safe=False,
)
