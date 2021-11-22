<div align="center">
    <br>
    <p align="center">
    <h1>MapReader</h1>
    </p>
    <h2>A computer vision pipeline for the semantic exploration of maps at scale</h2>
</div>
 
<p align="center">
    <a href="https://github.com/Living-with-machines/MapReader/workflows/Continuous%20integration/badge.svg">
        <img alt="Continuous integration badge" src="https://github.com/Living-with-machines/MapReader/workflows/Continuous%20integration/badge.svg">
    </a>
    <a href="./LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
    <br/>
</p>

MapReader is an end-to-end computer vision (CV) pipeline with two main components: preprocessing/annotation and training/inference.

<p align="center">
  <img src="./figs/MapReader_pipeline.png" 
        alt="MapReader pipeline" width="70%" align="center">
</p>

MapReader provides a set of tools to:

- **load** images/maps stored locally or **retrieve** maps via web-servers (e.g., tileserver which can be used to retrieve maps from OpenStreetMap *OSM* or National Library of Scotland *NLS*).
- **preprocess** images/maps (e.g., divide them into patches, resampling the images, removing borders outside the neatline or reprojecting the map).
- annotate images/maps or their patches (i.e. slices of an image/map) using an **interactive annotation tool**.
- **train, fine-tune, and evaluate** various CV models.
- **predict** labels (i.e., model inference) on large sets of images/maps.
- Other functionalities include:
    - various **plotting tools** using, e.g., *matplotlib*, *cartopy*, *Google Earth*, and [kepler.gl](https://kepler.gl/).
    - compute mean/standard-deviation **pixel intensity** of image patches.

Table of contents
-----------------

- [Installation and setup](#installation)
  - [Method 1: Anaconda + install dependencies manually](#method-1)
  - [Method 2: Anaconda + install dependencies using `environment.yaml` file](#method-2)
- [Tutorials](./examples) are organized in Jupyter Notebooks as follows:
  - Classification
      - [classification_one_inch_maps_001](./examples/classification_one_inch_maps_001)
        * **Goal:** train/fine-tune PyTorch CV classifiers on historical maps.
        * **Dataset:** from National Library of Scotland: [OS one-inch, 2nd edition layer](https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/index.html).
        * **Data access:** tileserver
        * **Annotations** are done on map patches (i.e., slices of each map).
        * **Classifier:** train/fine-tuned PyTorch CV models.

## Installation

We strongly recommend installation via Anaconda:

* Refer to [Anaconda website and follow the instructions](https://docs.anaconda.com/anaconda/install/).

### Method 1

* Create a new environment for `mapreader` called `mr_py38`:

```bash
conda create -n mr_py38 python=3.8
```

* Activate the environment:

```bash
conda activate mr_py38
```

* Clone `mapreader` source code:

```bash
git clone https://github.com/Living-with-machines/MapReader.git 
```

* Install `mapreader` dependencies:

```
# Install dependencies
conda install -c conda-forge rasterio
conda install shapely
conda install -c conda-forge notebook

pip install torch torchvision torchaudio
pip install tensorboard
pip install scikit-image
pip install timm
pip install "dask[complete]"
pip install geopandas
pip install tabulate
pip install joblib
pip install pyproj
pip install geopy
pip install aiohttp
pip install simplekml
pip install pytest
pip install git+https://github.com/kasra-hosseini/parhugin.git
```

For annotations, `mapreader` uses [ipyannotate](https://github.com/ipyannotate/ipyannotate) which can be installed from:

```
pip install ipyannotate
jupyter nbextension enable --py --sys-prefix ipyannotate
```

We use [kepler.gl](https://kepler.gl/) for visualization. To install:

```
pip install keplergl
jupyter nbextension install --py --sys-prefix keplergl 
jupyter nbextension enable --py --sys-prefix keplergl 
```

We also use [cartopy](https://scitools.org.uk/cartopy/docs/latest/installing.html) for plotting. It can be installed via:

```
conda install -c conda-forge cartopy
```

(Refer to [cartopy](https://scitools.org.uk/cartopy/docs/latest/installing.html) for more information.)

Only for users who want to use Azure: 

```
pip install azure-storage-blob
```

* Finally, install `mapreader` library:

```
cd /path/to/MapReader
pip install -v -e .
```

Alternatively:

```
cd /path/to/MapReader
python setup.py install
```

* We have provided some [Jupyter Notebooks to show how different components in `mapreader` work](./examples). 
  To allow the newly created `mr_py38` environment to show up in the notebooks:

```bash
python -m ipykernel install --user --name mr_py38 --display-name "Python (mr_py38)"
```

* Continue with the [Tutorials](#table-of-contents)!

### Method 2

* Create a new conda environment and install the dependencies using `environment.yaml` file:

```bash
conda env create --file environment.yaml python=3.8 --name mr_py38
```

* Activate the environment:

```bash
conda activate mr_py38
```

* For annotations, `mapreader` uses [ipyannotate](https://github.com/ipyannotate/ipyannotate) which can be installed from:

```
pip install ipyannotate
jupyter nbextension enable --py --sys-prefix ipyannotate
```

We use [kepler.gl](https://kepler.gl/) for visualization. To install:

```
pip install keplergl
jupyter nbextension install --py --sys-prefix keplergl 
jupyter nbextension enable --py --sys-prefix keplergl 
```

We also use [cartopy](https://scitools.org.uk/cartopy/docs/latest/installing.html) for plotting. It can be installed via:

```
conda install -c conda-forge cartopy
```

(Refer to [cartopy](https://scitools.org.uk/cartopy/docs/latest/installing.html) for more information.)


* Clone `mapreader` source code:

```bash
git clone https://github.com/Living-with-machines/MapReader.git 
```

* Finally, install `mapreader` library:

```
cd /path/to/MapReader
pip install -v -e .
```

Alternatively:

```
cd /path/to/MapReader
python setup.py install
```

* We have provided some [Jupyter Notebooks to show how different components in `mapreader` work](./examples). 
  To allow the newly created `mr_py38` environment to show up in the notebooks:

```bash
python -m ipykernel install --user --name mr_py38 --display-name "Python (mr_py38)"
```

* Continue with the [Tutorials](#table-of-contents)!