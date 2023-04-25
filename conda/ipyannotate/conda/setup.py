import setuptools

setuptools.setup(
    name="ipyannotate",
    version="0.1.0b0",
    description="Jupyter Widget for data annotation",
    author="Alexander Kukushkin",
    #author_email="",
    license="UNKNOWN",
    keywords=["ipython","jupyter","widgets"],
    zip_safe = False,
    url="https://github.com/alexanderkuk/ipyannotate",
    packages = setuptools.find_packages(),
    include_package_data = True,
    platforms="OS Independent",
    install_requires=[
        "ipywidgets>=7.0.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Framework :: IPython",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Graphics",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
    ],
)

