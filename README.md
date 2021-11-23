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

MapReader is an end-to-end computer vision (CV) pipeline with two main components: preprocessing/annotation and training/inference:

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
  - [Set up a conda environment](#set-up-a-conda-environment)
  - [Method 1: pip](#method-1)
  - [Method 2: poetry (for developers)](#method-2)
- [Tutorials](./examples) are organized in Jupyter Notebooks as follows:
  - Classification
      - [classification_one_inch_maps_001](./examples/classification_one_inch_maps_001)
        * **Goal:** train/fine-tune PyTorch CV classifiers on historical maps.
        * **Dataset:** from National Library of Scotland: [OS one-inch, 2nd edition layer](https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/index.html).
        * **Data access:** tileserver
        * **Annotations** are done on map patches (i.e., slices of each map).
        * **Classifier:** train/fine-tuned PyTorch CV models.
- [Credits and re-use terms](#credits-and-re-use-terms)

## Installation

### Set up a conda environment

We strongly recommend installation via Anaconda:

* Refer to [Anaconda website and follow the instructions](https://docs.anaconda.com/anaconda/install/).

* Create a new environment for `mapreader` called `mr_py38`:

```bash
conda create -n mr_py38 python=3.8
```

* Activate the environment:

```bash
conda activate mr_py38
```

### Method 1

* Install `mapreader`:

```bash
pip install git+https://github.com/Living-with-machines/MapReader.git
```

* Continue with the [Tutorials](#table-of-contents)!

### Method 2

* Clone `mapreader` source code:

```bash
git clone https://github.com/Living-with-machines/MapReader.git 
```

* Install using [poetry](https://python-poetry.org/):

```bash
cd /path/to/MapReader
poetry install
```

* Continue with the [Tutorials](#table-of-contents)!

## Credits and re-use terms

### Metadata

The re-use terms of the following metadata files stored in `mapreader/persistent_data` is CC-BY-NC-SA, restricting commercial re-use:

- `metadata_OS_One_Inch_GB_WFS.json`
- `metadata_OS_Six_Inch_GB_WFS.json`

Full details: https://maps.nls.uk/copyright.html#exceptions-os.