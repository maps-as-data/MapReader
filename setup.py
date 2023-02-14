import setuptools
import versioneer

EXTRAS_REQUIRE = {
        'geo': [
            "geopy==2.1.0",
            "rasterio>=1.2.10,<2.0.0",
            "keplergl>=0.3.2,<0.4.0",
            "simplekml>=1.3.6,<2.0.0"
            ],
        "dev": [
            "versioneer>=0.28",
        ]
}

setuptools.setup(
    name="mapreader",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="A computer vision pipeline for the semantic exploration of maps/images at scale",
    author=u"kasra-hosseini",
    #author_email="",
    license="MIT License",
    keywords=["Computer Vision", "Classification", "Deep Learning", "living with machines"],
    long_description = open('README.md', encoding="utf8").read(),
    long_description_content_type = 'text/markdown',
    zip_safe = False,
    url="https://github.com/Living-with-machines/MapReader",
    download_url="https://github.com/Living-with-machines/MapReader/archive/refs/heads/main.zip",
    packages = setuptools.find_packages(),
    include_package_data = True,
    platforms="OS Independent",
    python_requires='>=3.7',
    install_requires=[
        "pytest>=6.2.5,<7.0.0",
        "matplotlib>=3.5.0,<4.0.0",
        "numpy>=1.21.5,<2.0.0",
        "pandas>=1.3.4,<2.0.0",
        "pyproj>=3.2.0,<4.0.0",
        "azure-storage-blob>=12.9.0,<13.0.0",
        "aiohttp>=3.8.1,<4.0.0",
        "Shapely>=1.8.0,<2.0.0",
        "nest-asyncio>=1.5.1,<2.0.0",
        "scikit-image>=0.18.3,<0.19.0",
        "scikit-learn>=1.0.1,<2.0.0",
        "torch>=1.10.0,<2.0.0",
        "torchvision>=0.11.1,<0.12.1",
        "jupyter>=1.0.0,<2.0.0",
        "ipykernel>=6.5.1,<7.0.0",
        "ipyannotate==0.1.0-beta.0",
        "Cython>=0.29.24,<0.30.0",
        "proj>=0.2.0,<0.3.0",
        "PyYAML>=6.0,<7.0",
        "tensorboard>=2.7.0,<3.0.0",
        "parhugin>=0.0.3,<0.0.4"
	
    ],
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: OS Independent",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Sphinx~=6.1.3"],

    entry_points={
        'console_scripts': [
            'mapreader = mapreader.mapreader:main',
        ],
    }
)

