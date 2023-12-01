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
    <a href="https://zenodo.org/badge/latestdoi/430661738"><img src="https://zenodo.org/badge/430661738.svg" alt="DOI"></a>
    <br/>
</p>


## What is MapReader?

<div align="center">
    <figure>
    <img src="https://raw.githubusercontent.com/Living-with-machines/MapReader/main/figs/river_banner_8bit.png"
        alt="Annotated Map with Prediction Outputs"
        width="70%">
    </figure>
</div>

MapReader is an end-to-end computer vision (CV) pipeline for exploring and analyzing images at scale.

MapReader was developed in the [Living with Machines](https://livingwithmachines.ac.uk/) project to analyze large collections of historical maps but is a _**generalizable**_ computer vision pipeline which can be applied to _**any images**_ in a wide variety of domains.

## Overview

MapReader is a groundbreaking interdisciplinary tool that emerged from a specific set of geospatial historical research questions. It was inspired by methods in biomedical imaging and geographic information science, which were adapted for use by historians, for example in our [Journal of Victorian Culture](https://doi.org/10.1093/jvcult/vcab009) and [Geospatial Humanities 2022 SIGSPATIAL workshop](https://arxiv.org/abs/2111.15592) papers. The success of the tool subsequently generated interest from plant phenotype researchers working with large image datasets, and so MapReader is an example of cross-pollination between the humanities and the sciences made possible by reproducible data science.

### MapReader pipeline

<div align="center">
  <figure>
  <img src="https://raw.githubusercontent.com/Living-with-machines/MapReader/main/docs/source/figures/pipeline_explained.png"
        alt="MapReader pipeline"
        width="70%">
  </figure>
</div>

The MapReader pipeline consists of a linear sequence of tasks which, together, can be used to train a computer vision (CV) classifier to recognize visual features within maps and identify patches containing these features across entire map collections.

See our [About MapReader](https://mapreader.readthedocs.io/en/latest/About.html) page to learn more.

## Documentation

The MapReader documentation can be found at https://mapreader.readthedocs.io/en/latest/index.html.

**New users** should refer to the [Installation instructions](https://mapreader.readthedocs.io/en/latest/Install.html) and [Input guidance](https://mapreader.readthedocs.io/en/latest/Input-guidance.html) for help with the initial set up of MapReader.

**All users** should refer to our [User Guide](https://mapreader.readthedocs.io/en/latest/User-guide/User-guide.html) for guidance on how to use MapReader. This contains end-to-end instructions on how to use the MapReader pipeline, plus a number of worked examples illustrating use cases such as:

- Geospatial images (i.e. maps)
- Non-geospatial images

 **Developers and contributors** may also want to refer to the [API documentation](https://mapreader.readthedocs.io/en/latest/api/index.html) and [Contribution guide](https://mapreader.readthedocs.io/en/latest/Contribution-guide.html) for guidance on how to contribute to the MapReader package.

**Join our Slack workspace!**
Please fill out [this form](https://forms.gle/dXjECHZQkwrZ3Xpt9) to receive an invitation to the Slack workspace.

## What is included in this repo?

The MapReader package provides a set of tools to:

- **Download** images/maps and metadata stored on web-servers (e.g. tileservers which can be used to retrieve maps from OpenStreetMap (OSM), the National Library of Scotland (NLS), or elsewhere).
- **Load** images/maps and metadata stored locally.
- **Pre-process** images/maps:
  - patchify (create patches from a parent image),
  - resample (use image transformations to alter pixel-dimensions/resolution/orientation/etc.),
  - remove borders outside the neatline,
  - reproject between coordinate reference systems (CRS).
- **Annotate** images/maps (or their patches) using an interactive annotation tool.
- **Train or fine-tune** Computer Vision (CV) models and use these to **predict** labels (i.e. model inference) on large sets of images/maps.

Various **plotting and analysis** functionalities are also included (based on packages such as _matplotlib_, _cartopy_, _Google Earth_, and _[kepler.gl](https://kepler.gl/))_.

## How to cite MapReader

If you use MapReader in your work, please cite both the MapReader repo and [our SIGSPATIAL paper](https://dl.acm.org/doi/10.1145/3557919.3565812):

- Kasra Hosseini, Daniel C. S. Wilson, Kaspar Beelen, and Katherine McDonough. 2022. MapReader: a computer vision pipeline for the semantic exploration of maps at scale. In Proceedings of the 6th ACM SIGSPATIAL International Workshop on Geospatial Humanities (GeoHumanities '22). Association for Computing Machinery, New York, NY, USA, 8–19. https://doi.org/10.1145/3557919.3565812
- Kasra Hosseini, Rosie Wood, Andy Smith, Katie McDonough, Daniel C.S. Wilson, Christina Last, Kalle Westerling, and Evangeline Mae Corcoran. “Living-with-machines/mapreader: End of Lwm”. Zenodo, July 27, 2023. https://doi.org/10.5281/zenodo.8189653.


## Acknowledgements

This work was supported by Living with Machines (AHRC grant AH/S01179X/1) and The Alan Turing Institute (EPSRC grant EP/N510129/1).

Living with Machines, funded by the UK Research and Innovation (UKRI) Strategic Priority Fund, is a multidisciplinary collaboration delivered by the Arts and Humanities Research Council (AHRC), with The Alan Turing Institute, the British Library and the Universities of Cambridge, East Anglia, Exeter, and Queen Mary University of London.

Maps above reproduced with the permission of the National Library of Scotland https://maps.nls.uk/index.html

## Contributors

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-7-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://www.turing.ac.uk/people/researchers/katherine-mcdonough"><img src="https://avatars.githubusercontent.com/u/20363927?v=4?s=100" width="100px;" alt="Katie McDonough"/><br /><sub><b>Katie McDonough</b></sub></a><br /><a href="#research-kmcdono2" title="Research">🔬</a> <a href="#ideas-kmcdono2" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://danielwilson.info"><img src="https://avatars.githubusercontent.com/u/34318222?v=4?s=100" width="100px;" alt="Daniel C.S. Wilson"/><br /><sub><b>Daniel C.S. Wilson</b></sub></a><br /><a href="#research-dcsw2" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kasra-hosseini"><img src="https://avatars.githubusercontent.com/u/1899856?v=4?s=100" width="100px;" alt="Kasra Hosseini"/><br /><sub><b>Kasra Hosseini</b></sub></a><br /><a href="https://github.com/Living-with-machines/MapReader/commits?author=kasra-hosseini" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rwood-97"><img src="https://avatars.githubusercontent.com/u/72076688?v=4?s=100" width="100px;" alt="Rosie Wood"/><br /><sub><b>Rosie Wood</b></sub></a><br /><a href="https://github.com/Living-with-machines/MapReader/commits?author=rwood-97" title="Code">💻</a> <a href="https://github.com/Living-with-machines/MapReader/commits?author=rwood-97" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.westerling.nu"><img src="https://avatars.githubusercontent.com/u/7298727?v=4?s=100" width="100px;" alt="Kalle Westerling"/><br /><sub><b>Kalle Westerling</b></sub></a><br /><a href="https://github.com/Living-with-machines/MapReader/commits?author=kallewesterling" title="Code">💻</a> <a href="https://github.com/Living-with-machines/MapReader/commits?author=kallewesterling" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://maps.nls.uk"><img src="https://avatars.githubusercontent.com/u/3666702?v=4?s=100" width="100px;" alt="Chris Fleet"/><br /><sub><b>Chris Fleet</b></sub></a><br /><a href="#data-ChrisFleet" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kasparvonbeelen"><img src="https://avatars.githubusercontent.com/u/11618160?v=4?s=100" width="100px;" alt="Kaspar Beelen"/><br /><sub><b>Kaspar Beelen</b></sub></a><br /><a href="#ideas-kasparvonbeelen" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/Living-with-machines/MapReader/pulls?q=is%3Apr+reviewed-by%3Akasparvonbeelen" title="Reviewed Pull Requests">👀</a> <a href="#research-kasparvonbeelen" title="Research">🔬</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
