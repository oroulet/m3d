from setuptools import setup, find_packages

setup(
    name="m3d",
    version="0.3.0",
    description="Python3 3D transformation library with object oriented API and MIT licensed",
    author="Olivier Roulet-Dubonnet",
    author_email="olivier.roulet@gmail.com",
    url='https://gitlab.com/kurant-open/m3d',
    packages=find_packages(),
    provides=["m3d"],
    license="MIT",
    install_requires="numpy",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
