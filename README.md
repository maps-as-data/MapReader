<div align="center">
    <br>
    <p align="center">
    <h1>MapReader</h1>
    <h2>A computer vision pipeline for exploring and analyzing images at scale</h2>
    </p>
</div>
 
<p align="center">
    <a href="https://pypi.org/project/mapreader/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/MapReader">
    </a>
    <a href="https://github.com/Living-with-machines/MapReader/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
    <a href="https://github.com/Living-with-machines/MapReader/actions/workflows/mr_ci.yml/badge.svg">
        <img alt="Integration Tests badge" src="https://github.com/Living-with-machines/MapReader/actions/workflows/mr_ci.yml/badge.svg">
    </a>
    <br/>
</p>


## What is MapReader?

<!--- picture here? --->

MapReader is an end-to-end computer vision (CV) pipeline for exploring and analyzing images at scale. 

MapReader was developed in the [Living with Machines](https://livingwithmachines.ac.uk/) project to analyze large collections of historical maps but is a _**generalisable**_ computer vision pipeline which can be applied to _**any images**_ in a wide variety of domains. 


## Overview

MapReader is a groundbreaking interdisciplinary tool that emerged from a specific set of geospatial historical research questions. It was inspired by methods in biomedical imaging and geographic information science, which were adapted for annotation and use by historians, for example in [JVC](https://doi.org/10.1093/jvcult/vcab009) and [MapReader](https://arxiv.org/abs/2111.15592) papers. The success of the tool subsequently generated interest from plant phenotype researchers working with large image datasets, and so MapReader is an example of cross-pollination between the humanities and the sciences made possible by reproducible data science.

### MapReader pipeline 

The MapReader pipeline consists of two main components (See [Figure 2](#figure-2)):
1. preprocessing/annotation 
2. training/inference

<p align="center">
  <img src="https://raw.githubusercontent.com/Living-with-machines/MapReader/main/figs/MapReader_pipeline.png"
        alt="MapReader pipeline"
        width="70%">
</p>

<figcaption align="center">
  <b>Figure 2</b> - MapReader pipeline
</figcaption>
<br>

### What is included?

The MapReader package provides a set of tools to (See [Figure 1](#figure-1)):

- **download** images/maps and metadata stored on web-servers (e.g. tileserves which can be used to retrieve maps from OpenStreetMap (OSM), the National Library of Scotland (NLS), or elsewhere).
- **load** images/maps and metadata stored locally.
- **preprocess** images/maps:
  - patchify (create patches from a parent image), 
  - resample (use image transformations to alter pixel-dimensions/resolution/orientation/etc.),
  - remove borders outside the neatline,
  - reproject between coordinate reference systems (CRS).
- **annotate** images/maps (or their patches) using an interactive annotation tool.
- **train, fine-tune, and evaluate** Computer Vision (CV) models and use these to **predict** labels (i.e. model inference) on large sets of images/maps.

Various **plotting and analysis** functionalities are also included (based on packages such as *matplotlib*, *cartopy*, *Google Earth*, and [kepler.gl](https://kepler.gl/)).

<!--- or mb picture here? --->

## Documentation 

The MapReader documentation can be found [here](https://mapreader.readthedocs.io/en/latest/index.html).

**New users** should refer to the [Installation instructions](https://mapreader.readthedocs.io/en/latest/Install.html) and [Input guidance](https://mapreader.readthedocs.io/en/latest/Input-guidance.html) for guidance on the initial set up of MapReader.

**All users** should refer to our [User Guide](https://mapreader.readthedocs.io/en/latest/User-guide.html) for guidance on how to use MapReader. This contains end-to-end instructions on how to use the MapReader pipeline, plus a number of worked examples illustratng use cases such as:
- Geospatial images (i.e. maps)
- Non-geospatial images 

 **Developers and contributors** may also want to refer to the [API documentation](https://mapreader.readthedocs.io/en/latest/api/index.html) and [Contribution guide](https://mapreader.readthedocs.io/en/latest/Contribution-guide.html) for guidance on how to contribute to the MapReader package.

## How to cite MapReader

If you use MapReader in your work, please consider acknowledging us by citing [our paper](https://dl.acm.org/doi/10.1145/3557919.3565812):

- Kasra Hosseini, Daniel C. S. Wilson, Kaspar Beelen, and Katherine McDonough. 2022. MapReader: a computer vision pipeline for the semantic exploration of maps at scale. In Proceedings of the 6th ACM SIGSPATIAL International Workshop on Geospatial Humanities (GeoHumanities '22). Association for Computing Machinery, New York, NY, USA, 8–19. https://doi.org/10.1145/3557919.3565812


## Acknowledgements

This work was supported by Living with Machines (AHRC grant AH/S01179X/1) and The Alan Turing Institute (EPSRC grant EP/N510129/1). 

Living with Machines, funded by the UK Research and Innovation (UKRI) Strategic Priority Fund, is a multidisciplinary collaboration delivered by the Arts and Humanities Research Council (AHRC), with The Alan Turing Institute, the British Library and the Universities of Cambridge, East Anglia, Exeter, and Queen Mary University of London.