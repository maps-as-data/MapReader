# Changelog

## Python versions

The following table shows which versions of MapReader are compatible with which versions of python:

+----------+------------+
| Python   | MapReader  |
+==========+============+
| 3.9+     | v1.2.0     |
| 3.8-3.11 | <= v.1.1.5 |
+----------+------------+

---

## Pre-release

### Added

- Ability to install all three text-spotting frameworks at same time ([#514](https://github.com/maps-as-data/MapReader/pull/514))
- Can install text spotting dependencies using `pip install mapreader[text]` ([#514](https://github.com/maps-as-data/MapReader/pull/514))
- Text spotting code now covered by tests ([#514](https://github.com/maps-as-data/MapReader/pull/514))

## [v1.4.1](https://github.com/Living-with-machines/MapReader/releases/tag/v1.4.1) (2024-09-17)

### Changed

- Use `tqdm.auto` for progress bars

## [v1.4.0](https://github.com/Living-with-machines/MapReader/releases/tag/v1.4.0) (2024-09-12)

### Added

- `check_georeferencing` method and `georeferenced` attribute added to `MapImages` class ([#495](https://github.com/maps-as-data/MapReader/pull/495))
- `MapImages.convert_images` method now supports saving to GeoJSON format (set `save_format="geojson"`) ([#495](https://github.com/maps-as-data/MapReader/pull/495))
- All file loading methods now support `pathlib.Path` and `gpd.GeoDataFrame` objects as input ([#495](https://github.com/maps-as-data/MapReader/pull/495))
- Loading of dataframes from GeoJSON files now supported in many file loading methods (e.g. `add_metadata`, `Annotator.__init__`, `AnnotationsLoader.load`, etc.) ([#495](https://github.com/maps-as-data/MapReader/pull/495))
- `load_frames.py` added to `mapreader.utils`. This has functions for loading from various file formats (e.g. CSV, Excel, GeoJSON, etc.) and converting to GeoDataFrames ([#495](https://github.com/maps-as-data/MapReader/pull/495))
- Added tests for text spotting code ([#500](https://github.com/maps-as-data/MapReader/pull/500))
- Added `search_preds`, `show_search_results` and `save_search_results_to_geojson` methods to text spotting code ([#502](https://github.com/maps-as-data/MapReader/pull/502))

### Changed

- Refactoring of `SheetDownloader` to make full use of geopandas functionality - "metadata.json" is now read in as a GeoDataFrame instead of json dictionary ([#495](https://github.com/maps-as-data/MapReader/pull/495))
- `query_map_sheets_by_string` and `download_map_sheets_by_string` methods now search using columns in the GeoDataFrame instead of keys in the json dictionary ([#495](https://github.com/maps-as-data/MapReader/pull/495))
- `columns` argument renamed to `usecols` in `MapImages.add_metadata` method (to align with pandas) ([#495](https://github.com/maps-as-data/MapReader/pull/495))
- `polygon` column renamed to `geometry` (to align with geopandas) ([#495](https://github.com/maps-as-data/MapReader/pull/495))

### Removed

- `hist_published_dates` method removed from `SheetDownloader` as it is no longer needed. Use `sd.metadata["published_date"].hist()` instead ([#495](https://github.com/maps-as-data/MapReader/pull/495))

## [v.1.3.10](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.9) (2024-09-04)

_No changes to code. This release marks the move the the `maps-as-data` github organisation._

## [v1.3.9](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.9) (2024-08-21)

### Added

- Changelog added ([#461](https://github.com/Living-with-machines/MapReader/pull/461))

## Changes for versions <= v1.3.8 were added retroactively and may not be complete.

## [v1.3.8](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.8) (2024-08-12)

### Fixed

- Fixes the `get_label_index` method in `AnnotationsLoader` class ([#490](https://github.com/Living-with-machines/MapReader/pull/490))

## [v1.3.7](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.7) (2024-08-12)

### Added

- Adds option to show a border around central patch when annotating with context ([#480](https://github.com/Living-with-machines/MapReader/pull/480))
- Adds `show_vals` argument to show patch information when annotating ([#480](https://github.com/Living-with-machines/MapReader/pull/480))
- Adds ability to specify labels map when loading annotations ([#489](https://github.com/Living-with-machines/MapReader/pull/489))

## [v1.3.6](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.6) (2024-08-09)

### Added

- Adds new framework ([MapTextPipeline](https://github.com/rwood-97/MapTextPipeline/tree/main)) for running text spotting ([#474](https://github.com/Living-with-machines/MapReader/pull/474))
- Adds occlusion analysis to MapReader post processing ([#470](https://github.com/Living-with-machines/MapReader/pull/470))

### Changed

- Better error messaging [#476](https://github.com/Living-with-machines/MapReader/pull/476)

### Fixed

- Fixes [#442](https://github.com/Living-with-machines/MapReader/issues/442)

## [v1.3.5](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.5) (2024-08-05)

### Added

- Adds sphinx-autobuild for live updating of documentation ([#457](https://github.com/Living-with-machines/MapReader/pull/457))
- Adds data warning when downloading large amounts of data ([#464](https://github.com/Living-with-machines/MapReader/pull/464))

## [v1.3.4](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.4) (2024-07-05)

### Added

- Adds deduplication method to text spotting code [#435](https://github.com/Living-with-machines/MapReader/pull/435)
- Allows users to create overlapping patches [#435]((https://github.com/Living-with-machines/MapReader/pull/435))

## [v1.3.3](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.3) (2024-07-03)

### Changed

- Updates workshop notebooks ([#424](https://github.com/Living-with-machines/MapReader/pull/424))

## [v1.3.2](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.2) (2024-05-17)

### Added

- Adds CLI command to check installation and print MapReader version ([#431](https://github.com/Living-with-machines/MapReader/pull/431))
- Adds calculation of average across all channels for mean/std pixel stats ([#434](https://github.com/Living-with-machines/MapReader/pull/434))

### Changed

- Updates to worked examples and corresponding READMEs ([#414](https://github.com/Living-with-machines/MapReader/pull/414))

### Fixed

- Fixes [#400](https://github.com/Living-with-machines/MapReader/issues/400)
- Fixes [#430](https://github.com/Living-with-machines/MapReader/issues/430)

## [v1.3.1](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.1) (2024-05-03)

### Fixed

- Fixes [#398](https://github.com/Living-with-machines/MapReader/issues/398)

## [v1.3.0](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.0) (2024-04-29)

_This release was used for the Data/Culture Spring Workshop 2024._

### Added

- Adds text spotting to MapReader ([#388](https://github.com/Living-with-machines/MapReader/pull/388))

## [v1.2.0](https://github.com/Living-with-machines/MapReader/releases/tag/v1.2.0) (2024-04-08)

_This release removes support for Python 3.8._

### Added

- Support for Python 3.12 ([#384](https://github.com/Living-with-machines/MapReader/pull/384)

### Removed

- Support for Python 3.8 ([#384](https://github.com/Living-with-machines/MapReader/pull/384))

## [v1.1.5](https://github.com/Living-with-machines/MapReader/releases/tag/v1.1.5) (2024-04-04)

### Added

- Adds `download_in_parallel` argument to download methods ([#363](https://github.com/Living-with-machines/MapReader/pull/363))

### Removed

- Conda install option ([#366](https://github.com/Living-with-machines/MapReader/pull/366))

### Fixed

- Re-adds `square_cuts` argument to patchify images with deprecation warning ([#373](https://github.com/Living-with-machines/MapReader/pull/373))
- Fixes functions which reproject coordinates when converting between coordinate systems ([#374](https://github.com/Living-with-machines/MapReader/pull/374))

## [v1.1.4](https://github.com/Living-with-machines/MapReader/releases/tag/v1.1.4) (2024-02-23)

### Fixed

- Annotator no longer errors if "url" not in dataframe ([#357](https://github.com/Living-with-machines/MapReader/pull/357))

## [v1.1.3](https://github.com/Living-with-machines/MapReader/releases/tag/v1.1.3) (2024-02-21)

### Added

- Adds context-based post-processing script ([#342](https://github.com/Living-with-machines/MapReader/pull/342))
- Adds `save_predictions` method to the `ClassifierContainer` class ([#342](https://github.com/Living-with-machines/MapReader/pull/342))
- Adds `filter_for` argument to the `Annotator` class ([#319](https://github.com/Living-with-machines/MapReader/pull/319))
- Adds ability to create a `PatchContextDataset` from the `AnnotationsLoader` class ([#350](https://github.com/Living-with-machines/MapReader/pull/350))

### Changed

- The `ClassifierContainer` class now works with both `PatchDataset`s and `PatchContextDataset`s ([#350](https://github.com/Living-with-machines/MapReader/pull/350))
- Context classifier now works using a single branch model, one image, containing the patch and its context are used as input ([#350](https://github.com/Living-with-machines/MapReader/pull/350))

### Removed

- Removes `square_cuts` option when patchifying maps ([#350](https://github.com/Living-with-machines/MapReader/pull/350))

## [v1.1.2](https://github.com/Living-with-machines/MapReader/releases/tag/v1.1.2) (2024-02-05)

### Fixed

- Fixes [#346](https://github.com/Living-with-machines/MapReader/issues/346))

## [v1.1.1](https://github.com/Living-with-machines/MapReader/releases/tag/v1.1.1) (2024-01-08)

### Added

- Adds `add_coords_from_grid_bb` method to the `MapImages` class, allowing users to regenerate coordinates from grid bounding boxes
([#318](https://github.com/Living-with-machines/MapReader/pull/318))
- Adds `save_parents_as_geotiffs` method to the `MapImages` class, enabling users to save parent images as GeoTIFFs ([#318](https://github.com/Living-with-machines/MapReader/pull/318))
- Ensures CSV files are loaded correctly by applying `literal_eval` to all columns in the dataframe ([#318](https://github.com/Living-with-machines/MapReader/pull/318))
- Adds first draft of JOSS paper ([#321](https://github.com/Living-with-machines/MapReader/pull/321))

### Changed

- Annotator no longer requires `URL` column in parent dataframe ([#337](https://github.com/Living-with-machines/MapReader/pull/337))
- Annotator no longer accepts arbitrary kwargs ([#337](https://github.com/Living-with-machines/MapReader/pull/337))

### Fixed

- Fixes [#331](https://github.com/Living-with-machines/MapReader/issues/331)

## [v1.1.0](https://github.com/Living-with-machines/MapReader/releases/tag/v1.1.0) (2023-12-14)

### Fixed

- Fixes the `reproject_geo_info` method in `geo_utils` ([#317](https://github.com/Living-with-machines/MapReader/pull/317))

## [v1.0.7](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.7) (2023-12-14)

_New annotation interface!_

### Added

- Adds `all-contributors` bot to MapReader repo ([#297](https://github.com/Living-with-machines/MapReader/pull/297))

### Changed

- New annotation interface in MapReader ([#173](https://github.com/Living-with-machines/MapReader/pull/173))

## [v1.0.6](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.6) (2023-11-22)

_This release fixes a problem with calculating coordinates of downloaded maps. Releases prior to this will generate incorrect coordinates!_

### Added

- Adds `images_dir` argument when loading annotations, allowing users to specify the path to their patches ([#228](https://github.com/Living-with-machines/MapReader/pull/228))
- Adds pre-commit to the repo to ensure formatting is consistent ([#278](https://github.com/Living-with-machines/MapReader/pull/278))
- Adds option to go straight to inference in the `ClassifierContainer` class (e.g. with a already fine-tuned model) ([#280](https://github.com/Living-with-machines/MapReader/pull/280))
- Adds `metadata_to_save` and `data_col` arguments to the download methods, allowing users to specify additional metadata to save and the column to extract the published date from ([#291](https://github.com/Living-with-machines/MapReader/pull/291))

### Fixed

- Warn user if both lower left and upper right corners of the map are missing when downloading maps ([#269](https://github.com/Living-with-machines/MapReader/pull/269) and [#270](https://github.com/Living-with-machines/MapReader/pull/270))
- Fixes problem with calculating coordinates of downloaded maps ([#276](https://github.com/Living-with-machines/MapReader/pull/276))
- Fixes deduplication when downloading maps ([#285](https://github.com/Living-with-machines/MapReader/pull/285))

## [v1.0.5](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.5) (2023-07-25)

_This release marks the end of the Living with Machines project._

### Added

- State dicts are now saved when training/fine-tuning models ([#225](https://github.com/Living-with-machines/MapReader/pull/225))

### Changed

- Default delimiter for saving/loading files is now a comma (CSV) ([#241](https://github.com/Living-with-machines/MapReader/pull/241))

## No changelogs for versions < v1.0.5, see commit messages and PRs instead.

## [v1.0.4](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.4) (2023-06-27)

## Pull requests

- [#210](https://github.com/Living-with-machines/MapReader/pull/210)
- [#209](https://github.com/Living-with-machines/MapReader/pull/209)

## Commit messages

- save metadata to csv on each download
- add error message for broken image files

## [v1.0.3](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.3) (2023-06-21)

### Pull requests

- [#220](https://github.com/Living-with-machines/MapReader/pull/220)

### Commit messages

- adds details to dev docs about version numbers

## [v1.0.2](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.2) (2023-06-20)

### Pull requests

- [#200](https://github.com/Living-with-machines/MapReader/pull/200)
- [#197](https://github.com/Living-with-machines/MapReader/pull/197)
- [#202](https://github.com/Living-with-machines/MapReader/pull/202)

### Commit messages

- Update Input-guidance.rst w/NLS tile server details
- takes GH workflows from alto2txt
- removes poetry as build tool
- adds missing -m switch
- checkouts full git history for versioneer.py
- changes versioneer style
- fixes production PyPI deployment
- enables manual triggers

## [v1.0.1](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.1) (2023-06-06)

### Pull requests

- [#188](https://github.com/Living-with-machines/MapReader/pull/188)

### Commit messages

- initial commit for workshop notebook
- fix patchify by meters bug
- update workshop notebook and fix bug in images.py
- update notebook and create annotations file
- update workshop
- clear outputs and create 'exercise' notebook
- add show_parent() to workbook
- add annotations
- update workshop notebook w/ katies comments
- update empty notebook

## [v1.0.0](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.0) (2023-06-06)

### Pull requests

- [#95](https://github.com/Living-with-machines/MapReader/pull/95)
- [#140](https://github.com/Living-with-machines/MapReader/pull/140)
- [#155](https://github.com/Living-with-machines/MapReader/pull/155)
- [#151](https://github.com/Living-with-machines/MapReader/pull/151)
- [#165](https://github.com/Living-with-machines/MapReader/pull/165)
- [#160](https://github.com/Living-with-machines/MapReader/pull/160)
- [#163](https://github.com/Living-with-machines/MapReader/pull/163)
- [#180](https://github.com/Living-with-machines/MapReader/pull/180)
- [#154](https://github.com/Living-with-machines/MapReader/pull/154)
- [#176](https://github.com/Living-with-machines/MapReader/pull/176)
- [#164](https://github.com/Living-with-machines/MapReader/pull/164)
- [#182](https://github.com/Living-with-machines/MapReader/pull/182)
- [#185](https://github.com/Living-with-machines/MapReader/pull/185)
- [#199](https://github.com/Living-with-machines/MapReader/pull/199)
- [#195](https://github.com/Living-with-machines/MapReader/pull/195)

### Commit messages

- update tests
- fix coordinates assertion error
- ensure coordinates are xmin, ymin, xmax, ymax
- minor updaes to images.py
- Merge branch '82_align_task_names' into dev_load
- align to PEP8 style guide
- Merge branch '110_loader_file_paths' into dev_load
- Merge branch '116_data_inconsistencies' into dev_load
- update image_constructor
- update images_constructor
- update calc_pixel_width_height
- update calc_pixel_stats
- Update images.py    remove typo (double underscore infront of _add_shape_id)
- create parents and patches attributes
- edits to show/plotting methods
- move geoinfo method up + split for individual imgs
- update slicing methods (slicer now obsolete)
- edits to add_xx and calc_xxx
- rename/edit `add_par` to _add_patch_to_parent
- method to get tree_level from image_id
- method for verbose printing
- add patch coordinates
- auto pull in extra info when constructing images
- update init + imports
- run black
- updated test_loader_filepaths.py
- make resolving paths a separate static method
- update tests
- rename test files
- rename tests
- woops sorry - rename tests
- update tests and add option to ignore_mismatch in add_metadata
- fix assertions where keys are different
- rename test
- update tests for sheet_downloader
- fix tests
- Merge branch '110_loader_file_paths' into dev_load
- update tests
- add tests
- fix 'not a PNG file' error
- Merge branch '116_data_inconsistencies' into dev_load
- reorder to help with comparision to main
- update load_xx methods
- Merge branch '131_geotiff_bug' into dev_load
- fix rename of ``update`` to ``overwrite``
- fix tests
- move geo_utils to load subpackage and fix tests
- fix add metadata tests (checkout file from ``dev_load``)
- add developers guide to docs
- Update test_load_loader_add_metadata.py
- Update contribution guide into tutorials/docs/code
- update toc trees
- update toc trees - local only
- add tqdm and update show
- Add print info for where patches are saved
- minor fixes
- add tqdm to setup.py
- add beginners info + update input guidance
- allow metadata as excel file
- update tests with excel files
- add error if you try to load non-image files
- fix tmp_path not defined
- add beginners info text
- Update test_load.py
- add beginners info links
- save patches as geotiffs
- save patches as ".tif" only (not ".png.tif")
- final changes - include CRS as attribute
- update downloaders
- fix typo
- fix tests
- update saving of coordinates for metadata
- rename sample tif files
- update file names in tests
- rename tests
- add note about overwriting existing metadata info
- update reproject_geo_info
- make images_constructor private
- rename test_dirs to dirs
- add specific exception handling
- add tests for PIL.UnidentifiedImageError
- add tmp_paths
- rename proj2convert to target_crs
- merge test_images and test_images_add_metadata
- rename download/download2
- remove download_old
- update tests and init
- update extensions
- update load with advanced usage and to reflect updated code
- update load title
- ensure contribution guides included in sidebar
- add todos
- minor updates/ typo fixes:
- Update conf.py
- fix typo
- add 'save' argument to convert_images method
- fix tests
- Merge branch 'rw_docs' of https://github.com/Living-with-machines/MapReader into rw_docs
- update docs requirements and conf
- Update README.md
- Update Install.rst
- Update Install.rst
- Update README.md
- Update Beginners-info.rst    change title
- make csv default save as tab separated
- fix xxx_by_string() - allow no keys/keys as string
- update input guidance
- update download.rst (user guide)
- ensure CRS is retained in metadata.csv
- fix tests
- raise error if your coordinates are not 4326
- ensure dictionary keys are same for crs
- update env name (goodbye mr_py)
- fix typo
- Merge branch 'rw_docs' of https://github.com/Living-with-machines/MapReader into rw_docs
- ensure CRS in patch_df
- fix sample files
- update docstrings in dowloader_utils
- add grid_bb to/from polygon to downloader_utils
- fix ordering
- update error message
- set temp folder in tile_loading + import elsewhere
- add comment re. temp_folder
- fix grid_bb to/from polygon (remove self as arguments)
- update query_by_string tests
- add tests for crss
- fix tests
- fix crs key name
- add tests for saving dataframes
- Add tqdm to calc_pixel_stats
- Create test_sheet_downloader_mock_example.py
- minor fix to test_sheet_downloader_mock_example.py
- move load_annotations to learn
- rename 'kwds' to 'kwargs' to follow convention
- load_annotations now returns PatchDatasets
- save to geojson
- Merge branch '174-save-metadata' into dev_load
- save patches as geojson
- update docstrings and ensure PatchDataset methods work
- create new branch (mock_downloader) for mock downloader and rm file from this branch
- fix import tests
- fix tests
- fix PatchDataset in datasets.py
- add 'create_datasets' and 'create_dataloaders' methods to 'AnnotationsLoader'
- set up samplers by default in 'create_dataloader'
- add todo comment in train docs
- fix assignment of datasets variable
- update classifier class to ClassifierContainer
- add label indices to patch dataset
- update classifier
- fix classifier inference
- allow no label col/label indices in patchdataset
- add labels map to annotationsloader
- fix initialise model
- fix loading preexisting model
- update doc strings
- add torchinfo to requirements
- add tests
- make shapely version 2.0.0+
- keep label and pred columns if present when saving to geojson
- force geopandas to use shapely
- update tests
- fix patchify by meters bug
- rename to classify
- fix PatchContextDataset
- fix ClassfierContextContainer
- Update README.md
- Update README.md
- Merge branch 'rw_docs' of https://github.com/Living-with-machines/MapReader into rw_docs
- update classify name
- allow use of hugging face (tranformers) models
- update docs up to training part
- fix typos
- Update README.md    suggested removal of a line
- finish updating docs
- fix docs issues in classifier
- minor updates to pics
- add in-out pics
- fix unique_labels if using `append`
- add tests and fix deprecation  warning
- add tests for classifier
- update pkl files
- Update test_annotations_loader.py
- black
- formatting
- Merge branch 'dev_train' of https://github.com/Living-with-machines/MapReader into dev_train
- fix issue with load_patches (ensure patches are added to parents)
- make work with old annotate
- Update test_loader_load_patches.py
- Merge branch 'dev_train' of https://github.com/Living-with-machines/MapReader into dev_train
- fix some typos and add type hints
- Update setup.py
- minor fixes after testing
- update docs for saving geojson and csv files
- Merge branch 'rw_docs'
- fix to show_parent()
- updates from kmcdono2 comments
- add doc string and inference guidance
- fix train_test_split
- update venv env name from mr_py38 to mapreader
- update mnist worked example
- move small_mnist dataset
- move persistant data (NLS) dataset
- fix error if using mse loss
- ensure index has 'image_id' label when saved
- again fix issue with mse loss
- update plant pipeline
- update annotations files
- remove mnist old notebooks
- remove coastline example
- update classification_one_inch
- update annotation_tasks file
- update worked examples in docs
- Update README.md
- Update conf.py (turn off todos)

## [v0.3.4](https://github.com/Living-with-machines/MapReader/releases/tag/v0.3.4) (2023-02-20)

### Pull requests

- [#65](https://github.com/Living-with-machines/MapReader/pull/65)
- [#68](https://github.com/Living-with-machines/MapReader/pull/68)
- [#69](https://github.com/Living-with-machines/MapReader/pull/69)

### Commit messages

- version 0.3.3
- Add CI for pip install
- change links and citation from arxiv to ACM paper
- Update README.md
- initial commit
- initial commit
- initial commit
- adding .readthedocs.yaml
- add requirements.txt
- adding README.md
- update index.rst to include READMEs
- added READMEs
- fix requirements.txt
- setting up docs structure
- Update Load.rst
- Update Load.rst
- adding download section
- Create how_to_contribute_to_docs.md
- added basic docs to Download.rst using tutorial as base
- added basic docs to Load.rst using tutorial as base
- fix images in Load.rst
- added basic docs to Annotate.rst using tutorial as base
- organising docs files into separate directories per 'section'
- first thoughts
- updates to train.rst
- removed process/patchify and testing 'sphinx-disqus' extension
- added copybutton
- added copybutton + built html
- added basic docs to Train.rst using tutorial as base
- updated Load.rst after going through plant tutorial
- update Train.rst after going through plant tutorial
- updates to docs after testing
- Merge branch 'docs_train' into rw_docs
- updates to train.rst after testing
- Update Input-guidance.rst
- Add issue templates
- Add versioneer
- FIx proj and pytorch conda dependancy resolution
- tweak conda build tests
- build conda packages for upstream dependancies
- Successful local build of conda package

## [v0.3.3](https://github.com/Living-with-machines/MapReader/releases/tag/v0.3.3) (2022-04-27)

### Pull requests

- [#40](https://github.com/Living-with-machines/MapReader/pull/40)
- [#41](https://github.com/Living-with-machines/MapReader/pull/41)
- [#38](https://github.com/Living-with-machines/MapReader/pull/38)

### Commit messages

- Update README.md
- Move map-related libraries to extras_require
- Update setup.py
- Move rasterio to map dependencies
- Update README.md
- Update README.md
- :monocle_face: adding notebook to branch
- :bug: fixing empty notebook
- Update README.md
- Update README.md
- Update README
- Change version to 0.2.0
- Move maps notebooks to examples/maps
- Add MNIST example
- Add figure for MNIST tutorial
- Move plant notebooks to non-maps dir
- Update README
- Update README
- Update README
- Create README.md
- Update README.md
- Update README.md
- Create README.md
- Update README.md
- Update README.md
- Update README.md
- Update README.md
- Update README.md
- BUGFIX: annotation interface when working with images (and not patches)
- Update README.md
- Update README.md
- d test making apr    I've proposed removing 'originally' since we want to emphasise the ongoing nature of the library
- d actual edits to first para    I've restored the order to 'maps' -> 'images' so we get a clearer narative as in the current existing repo; and shortened / combined a sentence, as it was repeating 'non-maps' and 'maps', so I used 'any images' instead to make it more intuitive to read.  I was also going to add a few sentences giving the nice positive spin about interdisciplinary cross-pollination of image analysis, but not sure where this should go: I don't want to break the flow to the instructions, so perhaps it can go after the bullet points?
- Merge branch 'restructure-readme-kh' into patch-2
- minor
- Change map/non-map to geospatial/non-geospatial
- Change map/non-map to geospatial/non-geospatial
- Create README.md
- update README
- Merge branch 'restructure-readme-kh' of https://github.com/Living-with-machines/MapReader_public into restructure-readme-kh
- Move README to geospatial
- Update README
- Remove README from classification_one_inch_maps_001
- Add a new test
- Add MapReader paper
- Update README.md
- black
- black
- update CI
- update CI
- Add .[maps] to CI
- Remove as it is empty
- Version to 0.3.0
- update README
- update README
- update README
- update README
- Add windows-latest to CI
- README: geographic information science
- Move TOC after gallery
- Update README.md
- Add JVC paper
- Update README
- Update README
- Update README
- Update README
- Update README.md
- Set theme jekyll-theme-leap-day
- Update LICENSE
- Update README.md
- Update README.md
- Update README.md
- Update README.md
- Update README.md
- Update README
- Update README
- Update README.md
- Update README.md
- Update README
- Update README
- Update _config.yml
- Update README.md
- Update README.md
- Update README.md
- Update README.md
- Update README.md
- Set theme jekyll-theme-cayman
- Set theme jekyll-theme-minimal
- Update _config.yml
- Update README.md
- Update README.md
- Update README.md
- change maps to geo in installation
- review notebooks
- Add style
- Update README.md
- Update README.md
- Update README.md
- Update annotations for classification_one_inch_maps_001
- v0.3.1
- v0.3.2
- Update README.md
- update README
- add annotations
- 5 epochs

## [v0.1.2](https://github.com/Living-with-machines/MapReader/releases/tag/v0.1.2) (2022-03-03)

### Pull requests

- [#24](https://github.com/Living-with-machines/MapReader/pull/24)
- [#25](https://github.com/Living-with-machines/MapReader/pull/25)

### Commit messages

- :memo: adding additional kernel installation instructions.
- Merge remote-tracking branch 'origin/kasra-hosseini-patch-1' into adjust-poetry-installation-instructions
- Update README.md
- add requirements file
- update poetry files
- update requirements.txt
- update requirements.txt
- update requirements.txt
- merge with main
- :twisted_rightwards_arrows: merging branches
- import torchvision needed for show_sample method
- Add parhugin v0.0.3 to .toml file
- :package: adding requirements.txt file without hashes for binderhub build
- Merge branch 'dev' of https://github.com/Living-with-machines/MapReader into adjust-poetry-installation-instructions
- :package: adding   package for binderhub
- :package: removing  package for binderhub
- :pushpin: pinning usage: pyproj [-h] [-v] {sync} ...    pyproj version: 3.3.0 [PROJ version: 8.2.0]    optional arguments:    -h, --help     show this help message and exit    -v, --verbose  Show verbose debugging version information.    commands:    {sync} version to 3.2.1
- :pushpin: downgrading usage: pyproj [-h] [-v] {sync} ...    pyproj version: 3.3.0 [PROJ version: 8.2.0]    optional arguments:    -h, --help     show this help message and exit    -v, --verbose  Show verbose debugging version information.    commands:    {sync} to 3.2.1
- :package: adding  to
- :package: adding  and  file to branch
- Merge branch 'dev' into iss19-model-inference
- Add max_mean_pixel to the annotation interface
- Add setup.py, remove .lock and .toml files
- Update README
- udpate CI
- Update setup.py
- Update setup.py
- Add quick_start notebook
- Add files to quickstart notebook

## [v0.1.1](https://github.com/Living-with-machines/MapReader/releases/tag/v0.1.1) (2022-01-15)

The first published version of MapReader.
