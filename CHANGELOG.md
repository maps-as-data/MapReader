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

## [v1.3.8](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.8) (2024-08-12)

### Summary

The changelog discusses a fix for the `get_label_index` function due to an oversight in a previous commit. The ordering of unique labels no longer matched the labels in the labels map, so an inverse of the labels map is now utilized to determine the label index correctly. Additionally, tests have been added to verify this functionality. Code modifications were made in the `load_annotations.py` and `test_annotations_loader.py` files, while no files were removed. A checklist was completed before submitting the pull request, confirming self-review and test passes, though documentation updates were not included.

### Commit messages

- fix for get_label_index
- add tests

### Pull requests

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/490)
    > ### Summary
    > 
    > This was an oversight in #489 .
    > When getting the label indices from labels i was using the list of unique labels, but since the most recent changes the ordering of this list no longer reflects the ordering of labels in the labels map.
    > 
    > Now instead, we create an inverse of the labels map and use this to find the label index.
    > 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    > - [] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?

## Files

#### Modified

- mapreader/classify/load_annotations.py
- tests/test_classify/test_annotations_loader.py

## [v1.3.7](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.7) (2024-08-12)

### Summary

The changelog outlines several updates and fixes to the MapReader project. Key changes include the addition of a `border` argument to the annotator that creates outlines around patches when annotating, and a `show_vals` argument that allows users to display specific column information during annotation. Documentation has been updated accordingly, and comprehensive tests have been added and modified throughout the codebase.

Notable fixes include amendments to test cases and corrections to a reference citation in the bibliography. A new feature was introduced in the `AnnotationsLoader` to allow users to specify a labels map when loading annotations for consistent numbering across multiple rounds.

Overall, the updates enhance functionality, improve user documentation, and strengthen testing frameworks.

### Commit messages

- add border option when annotating
- fix tests
- add ability to show patch info when annotating
- add docs for `border`
- add docs for `show_vals`
- add and update tests
- add more tests
- Fix corcoran reference paper.bib
- add labels_map argument when loading annots
- add tests
- add docs

### Pull requests

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/480)
    > ### Summary
    > 
    > This adds a `border` argument to the annotator init and annotate methods, users can set this to `True` to add an outline to their patch if using context when annotating. e.g.
    > 
    > <img width="734" alt="image" src="https://github.com/user-attachments/assets/101953fb-32d1-4d8b-a4ee-b1dfef4cb97c">
    > 
    > It also adds a `show_vals` argument to the annotate method which allows them to specify a list of column names to show information about when annotating. e.g. 
    > <img width="607" alt="image" src="https://github.com/user-attachments/assets/de51210f-e7a9-4625-a217-ce3d8fab76e7">
    > 
    > Fixes #334 
    > Fixes #407 
    > 
    > ### Describe your changes
    > 
    > Describe changes/updates you have made to the code.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/489)
    > ### Summary
    > 
    > At the moment, when you load annotations, the `AnnotationsLoader` will auto-generate a labels map. This is useful for first round of annotations as you wont have one defined yet.
    > However, if you are doing a second round of annotations, you might want to use the same labels map as before to make sure the numbering is consistent. This option is implemented in this PR.
    > 
    > ### Describe your changes
    > 
    > This PR:
    > - Adds labels map argument when calling `load()` method of annotations loader. 
    > - Forces sorting of the labels map
    > - Implements method for appending new values to labels map. 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?

### Files

#### Modified

- docs/source/using-mapreader/step-by-step-guide/3-annotate.rst
- docs/source/using-mapreader/step-by-step-guide/4-classify/train.rst
- mapreader/annotate/annotator.py
- mapreader/classify/load_annotations.py
- paper/paper.bib
- tests/test_annotate/test_annotator.py
- tests/test_classify/test_annotations_loader.py


## [v1.3.6](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.6) (2024-08-09)

### Summary

The changelog summarizes various updates and modifications made to the MapReader project. Key changes include:

- **Minor Updates:** Adjustments to the software paper for the JOSS review, including formatting and readability improvements.
- **New Features:** Introduction of a fallback option in the code with a 5-second wait before retrying on errors, and the addition of an occlusion analysis code along with a new `OcclusionAnalyzer` class.
- **Testing Enhancements:** Numerous new tests added, including unit tests for different aspects of the software, and changes to existing test structures for better clarity.
- **Documentation Improvements:** Updates made to documentation, including clearer explanations and an overview of content rather than detailed page descriptions.
- **Code Refinements:** Various code improvements such as renaming variables, changing assertion statements to descriptive error messages, and enabling loading dataframes from paths.

The project also saw several pull requests addressing specific issues, including error handling and updating references in the documentation. Additionally, multiple files were added or modified, contributing to the overall enhancement of the software's functionality and usability.

### Commit messages

- Minor updates to software paper for JOSS review
- Adding a fallback option of waiting 5 seconds and trying again  Fixes #442
- Adding some spacing and explanatory comments
- add occlusion analysis code
- update tests
- add maptext runner
- add tests
- add formatting
- add sample railspace patch
- add more tests, make sure criterion is set when running occlusion
- add extra tests
- change criterion to loss_fn, add docs
- pre-commit fix
- enable loading dataframes from path
- add docs
- Replace most `assert` statements with more descriptive`raise` statements
- Update reference, remove suggested phrasing
- update init
- update init
- Update paper.bib
- fix indents
- Moving test files
- chore: Add unit tests for downloader_utils
- Renaming unrecognised test functions for download.data_structures
- Adding output to `_check_z` function
- Fix corcoran ref
- update docs section
- Update paper.md
- Update codecov versions
- Update paper.md

### Pull requests

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/474)
    > ### Summary
    > 
    > This PR adds a new framework for running text spotting to MapReader. 
    > 
    > The MapTextPipeline comes from the ICDAR map text competition. It is based on DeepSolo with a fine-tuned the David Rumsey map collection (from ICDAR). 
    > Our fork is here: https://github.com/rwood-97/MapTextPipeline/tree/main
    > Original is here: https://github.com/yyyyyxie/MapTextPipeline 
    > 
    > Model weights are linked in README.
    > 
    > ### Describe your changes
    > 
    > Adds a runner for the MapTextPipeline. 
    > This is now the one we should recommend people to use.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > ~- [ ] Add tests~ #403 
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [emdupre](https://github.com/Living-with-machines/MapReader/pull/446)
    > ### Summary
    > 
    > Relates to https://github.com/openjournals/joss-reviews/issues/6434
    > 
    > ### Describe your changes
    > 
    > Minor updates to the software paper accompanying the JOSS submission for MapReader for readability and formatting. 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [ ] Self-review code
    > - [ ] Ensure submission passes current tests
    > - [ ] Add tests
    > - [ ] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/482)
    > Update references for Corcoran and Li in JOSS paper
- [kallewesterling](https://github.com/Living-with-machines/MapReader/pull/476)
    > ### Summary
    > 
    > The `AssertionError` output isn't the most user-friendly response from software. In this PR, I've changed most of them into more helpful error messages. I'll also see if I can add tests for the exceptions, wherever I can manage to generate them.
    > 
    > Fixes #475
    > 
    > ### Describe your changes
    > 
    > Describe changes/updates you have made to the code.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [ ] Self-review code
    > - [ ] Ensure submission passes current tests
    > - [ ] Add tests
    > - [ ] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [kallewesterling](https://github.com/Living-with-machines/MapReader/pull/458)
    > ### Summary
    > 
    > @SWright2024 reported a problem where the removal of the temporary path caused a `PermissionError`. (Another possible error would be `OSError`.)
    > 
    > This PR creates a fallback option where there is a 5 second wait before trying again. If it's still an issue, the same error will be raised.
    > 
    > Fixes #442
    > 
    > ### Describe your changes
    > 
    > - [X] Adding in `time.sleep(5)` on `PermissionError`/`OSError`
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/485)
    > ### Summary
    > 
    > Since we updated the documentation it might be good to not explicitly explain each page in the docs but rather to give an overview of whats included.
    > This is towards https://github.com/openjournals/joss-reviews/issues/6434
    > 
    > Fixes #486 
    > 
    > ### Describe your changes
    > 
    > Changed the docs paragraph.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    > - [ ] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/470)
    > ### Summary
    > 
    > This adds occlusion analysis code to MapReader's post-processing methods.
    > 
    > Fixes #466 
    > Fixes #468 
    > 
    > ### Describe your changes
    > 
    > This PR:
    > - Adds `OcclusionAnalyzer` class to MapReader post processing
    > - Renames `PostProcessor` to `ContextPostProcessor`
    > - Renames `criterion` to `loss_fn`
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?

### Files

#### Added

- docs/source/_static/occlusion.png
- mapreader/process/occlusion_analysis.py
- mapreader/spot_text/maptext_runner.py
- tests/sample_files/patch-0-3045-145-3190-#map_100942121.png#.png
- tests/test_download/test_data_structures.py
- tests/test_download/test_downloader_utils.py
- tests/test_post_processing/test_occlusion.py

#### Modified

- .github/workflows/mr_ci.yml
- docs/requirements.txt
- docs/source/using-mapreader/step-by-step-guide/1-download.rst
- docs/source/using-mapreader/step-by-step-guide/4-classify/index.rst
- docs/source/using-mapreader/step-by-step-guide/4-classify/infer.rst
- docs/source/using-mapreader/step-by-step-guide/4-classify/train.rst
- docs/source/using-mapreader/step-by-step-guide/5-post-process.rst
- docs/source/using-mapreader/step-by-step-guide/6-spot-text.rst
- mapreader/__init__.py
- mapreader/annotate/utils.py
- mapreader/classify/classifier.py
- mapreader/classify/load_annotations.py
- mapreader/download/data_structures.py
- mapreader/download/downloader.py
- mapreader/download/downloader_utils.py
- mapreader/download/sheet_downloader.py
- mapreader/download/tile_merging.py
- mapreader/spot_text/deepsolo_runner.py
- mapreader/spot_text/dptext_detr_runner.py
- mapreader/spot_text/runner_base.py
- paper/paper.bib
- paper/paper.md
- setup.py
- tests/test_classify/test_classifier.py
- tests/test_geo_pipeline.py
- worked_examples/geospatial/classification_one_inch_maps/Pipeline.ipynb
- worked_examples/geospatial/context_classification_one_inch_maps/Pipeline.ipynb
- worked_examples/geospatial/workshop_june_2023/Workshop_June2023.ipynb
- worked_examples/geospatial/workshop_june_2023/Workshop_June2023_empty.ipynb
- worked_examples/geospatial/workshops_2024/Workshop_2024_part2.ipynb
- worked_examples/non-geospatial/classification_mnist/Pipeline.ipynb
- worked_examples/non-geospatial/classification_plant_phenotype/Pipeline.ipynb

## [v1.3.5](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.5) (2024-08-05)

### Summary

The changelog outlines various updates and modifications made to the MapReader documentation and codebase, focusing on improving organization, clarity, and usability. Key changes include:

1. **Documentation Enhancements**: 
   - Updated workshop tables and added sections on coding basics, a contribution guide, and user stories.
   - Moved and reformatted the "ways of working" document and incorporated `sphinxemoji` for better emoji support.
   - Splitted content into separate pages for clarity and added more consistent language throughout.

2. **File Management**:
   - Dropped unnecessary files and consolidated images and figures into designated directories.
   - Restructured documentation to improve navigation and SEO.

3. **Pre-commit Integration**: 
   - Added pre-commit and related documentation, ensuring developers are informed on installation and usage.

4. **Warnings and Error Handling**:
   - Introduced checks for large data downloads, adding warnings if more than 100MB is detected and allowing bypass through a 'force' argument.

5. **Pull Requests Overview**: 
   - Several pull requests were addressed, focusing on improving the structure and content of the documentation, fixing links, and resolving inconsistencies.

Overall, these commits and pull requests reflect an emphasis on making the MapReader project more accessible, user-friendly, and well-documented, alongside technical updates to enhance its functionality.

### Commit messages

- Updating tables for workshops
- This is not the correct filename (the correct file exists)
- `pre-commit` and `ruff` are currently not used
- moving `ways_of_working.md` to docs
- Moving ways of working and making rst
- Linking ways of working document
- Adding `sphinxemoji` for emoji support
- Fixing emoji
- Dropping these files as we don't need to render anything
- Moving `pre–commit` and `ruff` files back after @rwood-97 comment
- Add pre-commit to documentation  Fixes #453
- Updating a dysfunctional link
- Splitting up "beginners' info" into separate pages and adding more content
- Add `sphinx-autobuild` for ease of developing documentation  Fixes #456
- Adding a documentation paragraph as well
- add note about opening the url manually if needed
- Add documentation for reuse of mapreader  Fixes #294
- Editing language for consistency  Adding ``csv`` instead of csv, Python instead of python, and Jupyter instead of jupyter
- Removing TODO list from front page
- Moving toctree on front page
- Removing double messaging on front page
- Remodelling the documentation files
- Ensuring URL doesn't render in docs
- Removing saving of API files
- Dropping existing API files
- Changing to correct path in doctree
- Trying to rectify issue with H1 warnings in README ingestion into docs
- Moving style to test H1 alignment
- Dropping indices and tables from front page
- Add `api` files back in
- Simplifying README somewhat (also to get rid of Sphinx warnings)
- Dropping unnecessary API docs files from repo
- Removing new `api` docs from git commits
- Consolidating all figures into one directory
- Removing unused figures from root directory
- Adding `myst.header` suppression to remove warnings from MyST
- Moving all images to `_static` directory
- Changing README images to `_static` directory
- Add in sphinx include direction to front page
- Addresses #463
- Addresses #463
- Addresses #463
- Addresses #463
- Adding some easier access to other docs, addressing Add missing/edit ambiguous docstrings #463
- Adding more edits that addresses #463
- add warning for downloading large amounts of data (more than 100MB(?))
- update warnings
- update and add tests
- add note in docs
- fix download url
- Merge main and fix conflicts around coding basics
- Update test_sheet_downloader.py
- Committing the old contribution guide for easier moving forward
- Adding in "Share Your MapReader Story" to the new docs site

### Pull requests

- [kallewesterling](https://github.com/Living-with-machines/MapReader/pull/454)
    > Fixes #453
    > 
    > ### Summary
    > 
    > Adds in the pre-commit information in the developer documentation for MapReader.
    > 
    > Fixes #453
    > 
    > ### Describe your changes
    > 
    > - [ ] Added a block with `pre-commit` install and use information in the "developer's corner" of the documentation
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [kallewesterling](https://github.com/Living-with-machines/MapReader/pull/452)
    > ### Summary
    > 
    > I have cleaned up a little bit in the repository's files to make it look a bit cleaner.
    > 
    > ### Describe your changes
    > 
    > - [X] Removed `pre-commit` as we were not using it (we should?)
    > - [X] Removed `ruff` as we were not using it (we should)
    > - [X] moved the ways_of_working.md into the documentation, made it a RST
    > - [X] as a consequence of the above, I also added `spinxemoji` to be able to render one of the emoji's in the documentation.
- [kallewesterling](https://github.com/Living-with-machines/MapReader/pull/457)
    > ### Summary
    > 
    > Up until now, when rendering documentation, you have had to re-run the make html command every time. This makes it easier.
    > 
    > Fixes #456
    > 
    > ### Describe your changes
    > 
    > - [X] Adding in sphinx liveupdate
    > - [X] Creating make script
    > - [X] Adding developer documentation about this
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [kallewesterling](https://github.com/Living-with-machines/MapReader/pull/455)
    > ### Summary
    > 
    > With this PR, we want to finish up the Coding Basics section of the documentation.
    > 
    > Fixes #441
    > 
    > ### Describe your changes
    > 
    > - [X] ...
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/464)
    > ### Summary
    > 
    > This PR:
    > - adds code to work out how much data you will likely download when running any of the 'download' functions
    > - adds a warning if its over 100MB and raises an error
    > - adds a 'force' argument so that you can bypass this error
    > 
    > Fixes #444 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [kallewesterling](https://github.com/Living-with-machines/MapReader/pull/465)
    > There were a few more remaining references to the nls.uk site, which needed updating. I'm doing those here.
- [kallewesterling](https://github.com/Living-with-machines/MapReader/pull/459)
    > ### Summary
    > 
    > Added a section on the contribution guide for MapReader to contribute stories/case studies of how people are working with MapReader. This has made me think that we need a restructuring of the documentation pretty badly, as it's becoming quite unwieldy currently. But, at least we have this section now. Still remaining is to update tehe "Examples of User Stories" section with some real-world examples. @rwood-97 and @kmcdono2, can you think of any that would make good examples here?
    > 
    > Fixes #294
    > 
    > ### Describe your changes
    > 
    > Added a section on the contribution guide for MapReader to contribute stories/case studies of how people are working with MapReader.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [X] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > - [ ] Everything looks ok?
- [kallewesterling](https://github.com/Living-with-machines/MapReader/pull/462)
    > ### Summary
    > 
    > The documentation had unclear ordering and files that were named in ways that didn't optimise the SEO of our documentation. That should be fixed with this (see left image below, compared to the former structure on the right).
    > 
    > <img width="1797" alt="image" src="https://github.com/user-attachments/assets/c55cbef9-21fb-4f8d-b17a-351a760f7ffe">
    > 
    > Fixes #460
    > Fixes #463
    > 
    > ### Describe your changes
    > 
    > - minor edits of language for consistency
    > - fixing front page (removing the TODO list)
    > - remodelling the documentation files (moving quite a bit around!)
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [ ] Self-review code
    > - [ ] Ensure submission passes current tests
    > - [ ] Add tests
    > - [ ] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?

### Files

#### Added

- docs/source/coding-basics/index.rst
- docs/source/coding-basics/jupyter-notebooks.rst
- docs/source/coding-basics/python-packages.rst
- docs/source/coding-basics/terminal.rst
- docs/source/coding-basics/virtual-environments.rst
- docs/source/community-and-contributions/contribution-guide/developers-corner/index.rst
- docs/source/community-and-contributions/contribution-guide/developers-corner/installing-using-precommit.rst
- docs/source/community-and-contributions/contribution-guide/developers-corner/managing-version-numbers.rst
- docs/source/community-and-contributions/contribution-guide/developers-corner/running-tests.rst
- docs/source/community-and-contributions/contribution-guide/getting-started-with-contributions/index.rst
- docs/source/community-and-contributions/contribution-guide/how-to-add-yourself-as-a-contributor.rst
- docs/source/community-and-contributions/contribution-guide/index.rst
- docs/source/community-and-contributions/contribution-guide/ways-of-working.rst
- docs/source/community-and-contributions/events.rst
- docs/source/community-and-contributions/index.rst
- docs/source/community-and-contributions/joining-the-community.rst
- docs/source/community-and-contributions/share-your-story.rst
- docs/source/getting-started/index.rst
- docs/source/getting-started/installation-instructions/1-set-up-virtual-environment.rst
- docs/source/getting-started/installation-instructions/2-install-mapreader.rst
- docs/source/getting-started/installation-instructions/3-add-virtual-environment-to-notebooks.rst
- docs/source/getting-started/installation-instructions/index.rst
- docs/source/getting-started/troubleshooting-problems.rst
- docs/source/in-depth-resources/coding-basics/index.rst
- docs/source/in-depth-resources/coding-basics/jupyter-notebooks.rst
- docs/source/in-depth-resources/coding-basics/python-packages.rst
- docs/source/in-depth-resources/coding-basics/terminal.rst
- docs/source/in-depth-resources/coding-basics/virtual-environments.rst
- docs/source/in-depth-resources/index.rst
- docs/source/in-depth-resources/worked-examples/index.rst
- docs/source/in-depth-resources/worked-examples/non-geospatial-images.rst
- docs/source/introduction-to-mapreader/index.rst
- docs/source/introduction-to-mapreader/what-is-mapreader.rst
- docs/source/introduction-to-mapreader/what-skills-do-i-need-to-use-mapreader.rst
- docs/source/introduction-to-mapreader/who-might-be-interested-in-mapreader.rst
- docs/source/introduction-to-mapreader/why-should-you-use-mapreader.rst
- docs/source/using-mapreader/input-guidance/accessing-maps-via-tileservers.rst
- docs/source/using-mapreader/input-guidance/file-map-options.rst
- docs/source/using-mapreader/input-guidance/index.rst
- docs/source/using-mapreader/input-guidance/preparing-metadata.rst
- docs/source/using-mapreader/input-guidance/recommended-directory-structure.rst
- docs/source/using-mapreader/step-by-step-guide/index.rst

#### Modified

- .gitignore
- README.md
- docs/Makefile
- docs/requirements.txt
- docs/source/Contribution-guide/Contribution-guide.rst
- docs/source/conf.py
- docs/source/index.rst
- mapreader/__init__.py
- mapreader/annotate/annotator.py
- mapreader/classify/classifier.py
- mapreader/classify/datasets.py
- mapreader/classify/load_annotations.py
- mapreader/download/downloader.py
- mapreader/download/sheet_downloader.py
- mapreader/load/images.py
- mapreader/load/loader.py
- tests/test_geo_pipeline.py
- tests/test_sheet_downloader.py

#### Removed

- .all_contributors.rc
- _config.yml
- assets/css/style.scss
- docs/source/About.rst
- docs/source/Beginners-info.rst
- docs/source/Events.rst
- docs/source/Input-guidance.rst
- docs/source/Install.rst
- figs/MapReader_pipeline.png
- figs/mapreader_paper.png
- ways_of_working.md

## [v1.3.4](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.4) (2024-07-05)

### Summary

The changelog outlines recent updates and modifications to the codebase, emphasizing improvements related to deduplication and code organization. Key changes include:

1. **Code Enhancements**:
   - Added deduplication methods for handling overlapping polygons within and between patches.
   - Introduced a method for creating overlapping patches.
   - Established a base class for runners to streamline common functionalities.

2. **Documentation Updates**:
   - Edited documentation for community events and user guides.
   - Updated images related to the Intersection over Area (IoA).

3. **Testing and Fixes**:
   - Added tests for overlap functionality.
   - Fixed issues related to the IoA figure.

4. **General Maintenance**:
   - Removed placeholder text and revised several documentation files.

Multiple pull requests were made, including PR #435, which addresses overlapping polygons, improves code structure, and fixes several identified issues. The checklist confirms completion of self-review, testing, and documentation updates.

### Commit messages

- add deduplicate code to both runners
- move common functions to base class
- add deduplicate for parent images (e.g. if there is overlap between patches)
- add method for creating overlapping patches
- Editing upcoming communityevents
- Redoing list of community calls
- Removing placeholder text
- add parent deduplication
- Missing M
- update docs
- fix ioa figure
- add test for overlap

### Pull requests

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/435)
    > ### Summary
    > 
    > This PR adds new developments to the text spotting code:
    > - Adds deduplicate methods for removing overlapping polygons within a patch and also between patches
    > - Add overlap option when creating patches 
    > - Adds `Runner` base class to remove repeated code in the two different runners.
    > 
    > Fixes #404 
    > Fixes #405 
    > 
    > 
    > ### Describe your changes
    > 
    > Describe changes/updates you have made to the code.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Update relevant docs
    > - [x] Add tests where possible
    > Note: I haven't added tests for the deepsolo/dptext part because of installing detectron2, will make an issue about this
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?

### Files

#### Added

- docs/source/figures/IoA.png
- docs/source/figures/IoA_0.9.png
- mapreader/spot_text/runner_base.py

#### Modified

- docs/source/Events.rst
- docs/source/User-guide/Load.rst
- docs/source/User-guide/Spot-text.rst
- docs/source/index.rst
- mapreader/load/images.py
- mapreader/spot_text/deepsolo_runner.py
- mapreader/spot_text/dptext_detr_runner.py
- tests/test_load/test_images.py

## [v1.3.3](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.3) (2024-07-03)

### Summary

The changelog summarizes the following key updates:

#### Commit Messages
- Notebooks were updated and new ones were added for data visualization and text exploration.
- Documentation was improved, including aligning YOUR_TURNs with the code and adding references.
- Fixes included typos, workshop output handling, and ensuring compatibility with a new version of geopandas.
- Citations were updated, including the addition of a data/culture grant number to a paper.
- Miscellaneous updates to plots and maps, including improved clarity and filtering outputs.

#### Pull Requests
1. **PR #424**: Fixed various TODOs, added documentation links in notebooks, and aligned YOUR_TURNs with the code.
2. **PR #438**: Created a notebook for exploring text labels on maps for a June 2024 workshop.
3. **PR #443**: Updated author information in the paper.
4. **PR #449**: Added a citation for a relevant work to the JOSS paper.
5. **PR #450**: Mentioned the inclusion of text spotting in the related work of the paper.

#### Files
- Four new notebooks were added for various workshops and text exploration.
- Modifications were made to various documentation files, including updates to the user guide and paper citation files. 

Overall, the updates reflect ongoing improvements in documentation, functionality, and the addition of new educational materials.

### Commit messages

- udpate notebooks
- update YOUR_TURNs to align with docs
- add device argument to docs
- fix typo
- fix typo
- add printing f-scores per class
- add f-scores per class to docs
- rename workshops for 2024
- Add data/culture grant no. to paper
- add data viz notebooks
- small updates to plots
- update uk viz
- ignore workshop outputs
- ensure compatibility with geopands 1.0.0a1 pre-release
- update mapreader version
- add text exploration notebook
- add link to NLS website
- add reference for multilingual models
- Update Events.rst
- updates from Katie feedback
- clear outputs, filter for one parent map to save time, add a bit of text descriptions
- update June date
- fix workshop notebook part3
- Update authors paper.md
- update mapkurator citation
- paper update to mention text spotting
- corcoran update
- rosie's edit

### Pull requests

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/424)
    > ### Summary
    > 
    > This addresses some of the TODOs in #417 .
    > Fixes #420 
    > Fixes #411 
    > Fixes #408 
    > Fixes #409 
    > Fixes #406 
    > Fixes #410 
    > 
    > ### Describe your changes
    > 
    > Adds links to relevant documentation in the notebooks. 
    > 
    > TODO::
    > 
    >  - [x] Check each YOUR_TURN aligns with the code
    >  
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [x] Everything looks ok?
- [kasparvonbeelen](https://github.com/Living-with-machines/MapReader/pull/438)
    > ### Summary
    > 
    > Add a notebook for exploring and visualising text labels on maps for the June 2024 workshop in Lancaster
    > 
    > Fixes #<NUM>
    > 
    > ### Describe your changes
    > 
    > Describe changes/updates you have made to the code.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [ ] Self-review code
    > - [ ] Ensure submission passes current tests
    > - [ ] Add tests
    > - [ ] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/443)
    > Update authors
- [kmcdono2](https://github.com/Living-with-machines/MapReader/pull/449)
    > ### Summary
    > 
    > Addresses citation update request (https://github.com/openjournals/joss-reviews/issues/6434#issuecomment-2184173141) for JOSS paper from @emdupre
    > 
    > 
    > ### Describe your changes
    > 
    > Added citation for mapkurator:
    > 
    > @inproceedings{li2020automatic,
    >   title={An automatic approach for generating rich, linked geo-metadata from historical map images},
    >   author={Li, Zekun and Chiang, Yao-Yi and Tavakkol, Sasan and Shbita, Basel and Uhl, Johannes H and Leyk, Stefan and Knoblock, Craig A},
    >   booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
    >   pages={3290--3298},
    >   year={2020}
    > }
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [ ] Self-review code
    > - [ ] Ensure submission passes current tests
    > - [ ] Add tests
    > - [ ] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [x] Everything looks ok?
- [kmcdono2](https://github.com/Living-with-machines/MapReader/pull/450)
    > ### Summary
    > 
    > Added sentence to related work to mention briefly that MapReader now includes text spotting.
    > 
    > ### Describe your changes
    > 
    > N/A
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [ ] Self-review code
    > - [ ] Ensure submission passes current tests
    > - [ ] Add tests
    > - [ ] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?

### Files

#### Added

- worked_examples/geospatial/workshops_2024/Workshop_2024_part1.ipynb
- worked_examples/geospatial/workshops_2024/data_viz_all_uk.ipynb
- worked_examples/geospatial/workshops_2024/data_viz_small.ipynb
- worked_examples/geospatial/workshops_2024/explore_text_on_maps.ipynb

#### Modified

- .gitignore
- docs/source/Events.rst
- docs/source/User-guide/Classify/Infer.rst
- docs/source/User-guide/Classify/Train.rst
- paper/paper.bib
- paper/paper.md

## [v1.3.2](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.2) (2024-05-17)

### Summary

The changelog details several updates and fixes made to the MapReader project, primarily focusing on dependencies, code functionality, and documentation improvements. Key points are:

- **Code Updates**: Several scripts and modules were updated, including `post_process.py`, model names, and various annotator functionalities to resolve warnings and errors.
- **Dependency Management**: Minimum versions of dependencies such as `joblib` were updated.
- **Documentation**: Multiple README files and examples were restructured and updated for clarity, including adding instructions on how to run tests and clarifying geospatial and non-geospatial resources.
- **Bug Fixes**: Various bugs were addressed, including model weights warnings, command line script issues, and label review processes.
- **New Features**: Enhanced file handling for annotations and pixel statistics calculations were introduced.
- **File Changes**: New files were added, including the main script and annotation examples, while some outdated examples were removed.

Overall, this update streamlines the code, enhances usability through documentation, and fixes various issues to improve the project’s functionality.

### Commit messages

- update deps
- update post_process.py
- update minimum joblib
- fix model weights warning
- fix post processing tests
- update timm model names
- remove re.search
- update annotator to fix warnings
- new README structure
- add how to run tests
- Update README.md
- Update README.md
- Update README.md
- Update README.md
- Update README.md
- Update non-geospatial README.md
- Update README.md
- Update README.md
- Update README.md
- Update Worked-examples.rst
- Update Worked-examples.rst
- Update Worked-examples.rst
- Update Download.rst
- move annotation worked examples
- remove postproc worked example
- update geospatial readme table
- update instructions for worked examples
- update example naming
- remove annotations col from readme
- updates csv/tsv
- update test instructions
- update y-labels in metric plots
- fix command line script
- fix review_labels
- fix add_id bug
- update pixel stats calculation

### Pull requests

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/413)
    > ### Summary
    > 
    > Fixes #398 
    > 
    > ### Describe your changes
    > 
    > e.g. https://maps.nls.uk/view/101437802 goes over the boundary and so is registered twice in the NLS metadata but with two sets of coordinates. 
    > Instead of erroring or overwriting the sheet, you will now get map_xxxx.png and map_xxxx_1.png saved and these will have separate metadata so separate coordinates.
    > 
    > This also adds a mock function when downloading maps and so fixes #183 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/396)
    > ### Summary
    > 
    > Fixes #385 
    > 
    > ### Describe your changes
    > 
    > Update code and dependencies to ensure fix warnings in tests
    > 
    > To do:
    > - [x] Update docs re. how to run tests
    > - [x] Deal with annotator 
    > - [ ] Zero division error ?
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [kmcdono2](https://github.com/Living-with-machines/MapReader/pull/416)
    > ### Summary
    > 
    > Fixes #389 
    > 
    > ### Describe your changes
    > 
    > - updated all typos mentioned in https://github.com/openjournals/joss-reviews/issues/6434#issuecomment-2041626766
    > 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [kmcdono2](https://github.com/Living-with-machines/MapReader/pull/414)
    > ### Summary
    > 
    > Addresses "update geospatial README" in https://github.com/orgs/Living-with-machines/projects/7/views/1?pane=issue&itemId=59167565 
    > 
    > Fixes part of  #<386>
    > 
    > ### Describe your changes
    > 
    > Changed README from content that is now mostly in documentation site to clear structure for list of worked examples and required files.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [kmcdono2](https://github.com/Living-with-machines/MapReader/pull/415)
    > ### Summary
    > 
    > Fixes non-geospatial README issues in #<386>
    > 
    > ### Describe your changes
    > 
    > Reformatted to standard fields per worked example, matching geospatial README.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/418)
    > Addresses #386 
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/432)
    > ### Summary
    > 
    > Fixes #400 
    > 
    > ### Describe your changes
    > 
    > Previous code allowed you enter a new label when reviewing labels but would give you an error (even though it relabelled the patch fine). Now we give an error and only allow user to enter one of the existing labels when re-labelling/reviewing their annotations.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    > - [ ] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/431)
    > ### Summary
    > 
    > Fixes #64 
    > 
    > ### Describe your changes
    > 
    > Command line call now prints mapreader version
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    > - [ ] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/433)
    > ### Summary
    > 
    > Fixes #430 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    > - [ ] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/434)
    > ### Summary
    > 
    > Adds calculation of average across all channels for mean/std pixel stats.
    > Also makes sure image cropping is working for calc pixel stats.
    > Fixes #426 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    > - [ ] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?

### Files

#### Added

- mapreader/__main__.py
- worked_examples/geospatial/workshop_june_2023/annotations/buildings_#workshop#.tsv
- worked_examples/non-geospatial/classification_plant_phenotype/annotations_phenotype_open_access/phenotype_test_#kasra#.tsv

#### Modified

- docs/source/Install.rst
- docs/source/User-guide/Download.rst
- docs/source/User-guide/Post-process.rst
- docs/source/Worked-examples/Worked-examples.rst
- mapreader/__init__.py
- mapreader/annotate/annotator.py
- mapreader/classify/classifier.py
- mapreader/classify/load_annotations.py
- mapreader/download/sheet_downloader.py
- mapreader/load/images.py
- mapreader/process/post_process.py
- setup.py
- tests/test_annotator.py
- tests/test_classify/test_classifier.py
- tests/test_post_processing.py
- tests/test_sheet_downloader.py
- worked_examples/geospatial/README.md
- worked_examples/geospatial/classification_one_inch_maps/Pipeline.ipynb
- worked_examples/geospatial/workshop_june_2023/Workshop_June2023.ipynb
- worked_examples/geospatial/workshop_june_2023/annotations/buildings_#workshop#.csv
- worked_examples/non-geospatial/README.md
- worked_examples/non-geospatial/classification_mnist/Pipeline.ipynb
- worked_examples/non-geospatial/classification_plant_phenotype/Pipeline.ipynb
- worked_examples/non-geospatial/classification_plant_phenotype/annotations_phenotype_open_access/phenotype_test_#kasra#.csv

#### Removed

- worked_examples/postproc/postproc_compute_rail_density.ipynb

## [v1.3.1](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.1) (2024-05-03)

### Summary

The changelog presents a series of updates to a project, which includes the addition of new notebooks related to workshops, specifically for April/May 2024 and a June 2023 notebook. Key changes include:

- **New Notebooks Added**: Several new notebooks for the April/May 2024 workshop were created, including deep learning and metadata exploration notebooks.
- **Notebooks Updated**: Existing notebooks were updated based on comments and for improvements, including fixing issues related to saving and overwriting.
- **Code Fixes**: Issues such as a problem with `patches_to_geojson` and saving for duplicate sheet names were addressed.
- **Additional Features**: Version information was added to notebooks, and a split for inference-only notebooks was introduced.
- **File Modifications**: Certain scripts related to downloading and merging tiles and tests were modified.

A related pull request was made to facilitate these changes, focusing on the new workshop notebooks. Overall, the updates aim to enhance the usability and functionality of the tools provided in the project.

### Commit messages

- add version info to june 2023 notebook
- add 2024 workshop notebook
- update notebook
- split notebooks and add inference only
- add deepsolo notebook
- address comments in part1 and extra
- update part2
- fix patches_to_geojson problem
- add metadata notebook
- change other patches to 1000 meters
- load 1000 pixel patdches
- update version
- update deepsolo notebook
- update saving for duplicate sheet names
- fix overwriting issue
- update tests

### Pull requests

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/399)
    > ### Summary
    > 
    > Adding notebooks for the April/May workshop.
    > 
    > Fixes #394 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?

### Files

#### Added

- worked_examples/geospatial/workshop_april_2024/Workshop_AprilMay2024_DeepSolo.ipynb
- worked_examples/geospatial/workshop_april_2024/Workshop_AprilMay2024_extra.ipynb
- worked_examples/geospatial/workshop_april_2024/Workshop_AprilMay2024_part1.ipynb
- worked_examples/geospatial/workshop_april_2024/Workshop_AprilMay2024_part2.ipynb
- worked_examples/geospatial/workshop_april_2024/metadata_exploration.ipynb

#### Modified

- mapreader/download/sheet_downloader.py
- mapreader/download/tile_merging.py
- tests/sample_files/test_json.json
- tests/test_sheet_downloader.py
- worked_examples/geospatial/workshop_june_2023/Workshop_June2023_empty.ipynb

## [v1.3.0](https://github.com/Living-with-machines/MapReader/releases/tag/v1.3.0) (2024-04-29)

### Summary

The changelog outlines various updates and improvements made to the MapReader project, including the addition of a CHANGELOG file, updates to installation instructions, and enhancements to text detection features. Key changes include:

- **New Features:** 
  - Introduction of `DeepSoloRunner` and `DPTextDETRRunner` classes for running respective models on images, facilitating text detection and recognition.
  - A new method `run_all` allows users to operate on all patches in a provided patch DataFrame.

- **Documentation Updates:**
  - The changelog now includes a Python version compatibility table and updated installation instructions.
  - Documentation has been added or updated for text spotting and user guides.

- **Dependency Management:** 
  - Cartopy has been updated to a required dependency to simplify installation.
  
- **Refactoring and Fixes:**
  - Various typos and import issues were resolved, and methods were updated or renamed for clarity.
  - Improved error handling and updating of file-saving documentation.

- **Testing Enhancements:** 
  - Testing has been added for Python 3.12, and relevant documents were updated accordingly.

- **Community Updates:** 
  - An April community call summary was noted.

The commit messages and pull requests highlight collaborative efforts in refining the codebase and expanding features, particularly around text detection and model integration.

### Commit messages

- add changelog
- update changelog with python version table
- update installation instructions
- add deepsolo runner
- add run all method
- update for if patch_df not passed
- Update publish-to-test-pypi.yml    only run on main branch
- fix typo in show
- add show method
- add to imports
- allow pass on import of DeepSoloRunner
- add worked example
- rename to allow for different runners
- fix init
- fix imports, add run_all arg to dptext detr runner
- update worked examples
- April community call update
- add build to git ignore
- rename text_spot to spot_text
- update file saving docs
- add docs for spotting text
- add file to user guide toc tree
- fix errors
- Update README.md
- Update README.md
- update install docs to use 3.1-0
- update filepath
- update dptext detr runner import
- update docs for install
- update where to find configs and weights

### Pull requests

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/384)
    > ### Summary
    > 
    > Addresses #229
    > Fixes #364
    > 
    > ### Describe your changes
    > 
    > This PR:
    > - Updates supported python versions in MapReader from 3.8-3.11 to 3.9+
    > - Adds cartopy as required dependency vs optional as 0.22 is simpler to install
    > - Updates docs accordingly
    > - Adds testing for python 3.12
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [kmcdono2](https://github.com/Living-with-machines/MapReader/pull/395)
    > ### Summary
    > 
    > Describe the problem you're trying to fix in this pull request.
    > Please reference any related issue and use fixes/close to automatically close them, if pertinent. For example: "Fixes #58", or "Addresses (but does not close) #238". -->
    > 
    > Fixes #<NUM>
    > 
    > ### Describe your changes
    > 
    > Describe changes/updates you have made to the code.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [ ] Self-review code
    > - [ ] Ensure submission passes current tests
    > - [ ] Add tests
    > - [ ] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/388)
    > ### Summary
    > 
    > This PR begins the work of bringing text detection/recognition (ie. text spotting) into MapReader.
    > 
    > Addresses #358
    > 
    > ### Describe your changes
    > 
    > `DeepSoloRunner` is a class used for running the [DeepSolo model](https://github.com/ViTAE-Transformer/DeepSolo/tree/main) on images. This produces polygons + text.
    > `DPTextDETRRunner` is a class used for running the [DPText-DETR model](https://github.com/ymy-k/DPText-DETR) on images.  This produces just polygons.
    > 
    > To do:
    > 
    > - [x] Allow user to run on all patches in provided patch df
    > - [x] Add plotting function
    > - [x] Add docs for getting this running
    > - [x] Check licences
    > - [x] Add worked example
    > - [ ] Add polygon merging code (?)
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    > - [ ] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?

### Files

#### Added

- CHANGELOG.md
- docs/source/User-guide/Spot-text.rst
- mapreader/spot_text/__init__.py
- mapreader/spot_text/deepsolo_runner.py
- mapreader/spot_text/dptext_detr_runner.py
- worked_examples/geospatial/text_spotting_one_inch_maps/deepsolo/Pipeline.ipynb
- worked_examples/geospatial/text_spotting_one_inch_maps/deepsolo/README.md
- worked_examples/geospatial/text_spotting_one_inch_maps/dptext-detr/Pipeline.ipynb
- worked_examples/geospatial/text_spotting_one_inch_maps/dptext-detr/README.md

#### Modified

- .github/workflows/publish-to-test-pypi.yml
- .gitignore
- docs/source/Events.rst
- docs/source/Install.rst
- docs/source/User-guide/Classify/Infer.rst
- docs/source/User-guide/Classify/Train.rst
- docs/source/User-guide/User-guide.rst
- mapreader/__init__.py
- mapreader/load/images.py
- worked_examples/non-geospatial/README.md

## [v1.2.0](https://github.com/Living-with-machines/MapReader/releases/tag/v1.2.0) (2024-04-08)

### Summary

The changelog details updates made primarily by Dependabot, focusing on library dependency upgrades, configuration changes, and workflow additions. Key updates include:

1. **Dependency Updates:**
   - `pytest` updated to allow versions `<9.0.0`.
   - `black` requirement updated to allow versions `>=23.7.0,<25.0.0`.
   - `torchvision` updated to support versions `>=0.11.1,<0.17.3`.
   - `flake8` updated to allow versions `>=6.0.0,<8.0.0`.
   - `pytest-cov` requirement updated to allow versions `>=4.1.0,<6.0.0`.

2. **File Changes:**
   - Created `.github/dependabot.yml` and added a Dependabot review workflow.
   - Various modifications to workflow files, requirements, and source code files.

3. **General Improvements:**
   - Updated allowed Python versions and the version referenced in project files.
   - Moved `cartopy` to required dependencies and updated its version.

Overall, the commit messages indicate continuous maintenance to ensure compatibility with the latest versions of dependencies while enhancing project workflows.

### Commit messages

- Create dependabot.yml
- Update pytest requirement from <8.0.0 to <9.0.0    Updates the requirements on [pytest](https://github.com/pytest-dev/pytest) to permit the latest version.  - [Release notes](https://github.com/pytest-dev/pytest/releases)  - [Changelog](https://github.com/pytest-dev/pytest/blob/main/CHANGELOG.rst)  - [Commits](https://github.com/pytest-dev/pytest/compare/1.0.0b3...8.1.1)    ---  updated-dependencies:  - dependency-name: pytest    dependency-type: direct:development  ...    Signed-off-by: dependabot[bot] <support@github.com>
- Update black requirement from <24.0.0,>=23.7.0 to >=23.7.0,<25.0.0    Updates the requirements on [black](https://github.com/psf/black) to permit the latest version.  - [Release notes](https://github.com/psf/black/releases)  - [Changelog](https://github.com/psf/black/blob/main/CHANGES.md)  - [Commits](https://github.com/psf/black/compare/23.7.0...24.3.0)    ---  updated-dependencies:  - dependency-name: black    dependency-type: direct:development  ...    Signed-off-by: dependabot[bot] <support@github.com>
- Update torchvision requirement from <0.12.1,>=0.11.1 to >=0.11.1,<0.17.3    Updates the requirements on [torchvision](https://github.com/pytorch/vision) to permit the latest version.  - [Release notes](https://github.com/pytorch/vision/releases)  - [Commits](https://github.com/pytorch/vision/compare/v0.11.1...v0.17.2)    ---  updated-dependencies:  - dependency-name: torchvision    dependency-type: direct:production  ...    Signed-off-by: dependabot[bot] <support@github.com>
- Update dependabot.yml
- Update dependabot.yml
- Update flake8 requirement from <7.0.0,>=6.0.0 to >=6.0.0,<8.0.0    Updates the requirements on [flake8](https://github.com/pycqa/flake8) to permit the latest version.  - [Commits](https://github.com/pycqa/flake8/compare/6.0.0...7.0.0)    ---  updated-dependencies:  - dependency-name: flake8    dependency-type: direct:development  ...    Signed-off-by: dependabot[bot] <support@github.com>
- Update pytest-cov requirement from <5.0.0,>=4.1.0 to >=4.1.0,<6.0.0    Updates the requirements on [pytest-cov](https://github.com/pytest-dev/pytest-cov) to permit the latest version.  - [Changelog](https://github.com/pytest-dev/pytest-cov/blob/master/CHANGELOG.rst)  - [Commits](https://github.com/pytest-dev/pytest-cov/compare/v4.1.0...v5.0.0)    ---  updated-dependencies:  - dependency-name: pytest-cov    dependency-type: direct:development  ...    Signed-off-by: dependabot[bot] <support@github.com>
- update allowed python versions
- update python version in files
- add dependabot review workflow
- remove cap on torch version
- force int for randint
- move cartopy to required and update version
- update docs

### Pull requests

- [dependabot[bot]](https://github.com/Living-with-machines/MapReader/pull/379)
    > Updates the requirements on [pytest](https://github.com/pytest-dev/pytest) to permit the latest version.
    > <details>
    > <summary>Release notes</summary>
    > <p><em>Sourced from <a href="https://github.com/pytest-dev/pytest/releases">pytest's releases</a>.</em></p>
    > <blockquote>
    > <h2>8.1.1</h2>
    > <h1>pytest 8.1.1 (2024-03-08)</h1>
    > <p>::: {.note}
    > ::: {.title}
    > Note
    > :::</p>
    > <p>This release is not a usual bug fix release -- it contains features and improvements, being a follow up
    > to <code>8.1.0</code>, which has been yanked from PyPI.
    > :::</p>
    > <h2>Features</h2>
    > <ul>
    > <li>
    > <p><a href="https://redirect.github.com/pytest-dev/pytest/issues/11475">#11475</a>: Added the new <code>consider_namespace_packages</code>{.interpreted-text role=&quot;confval&quot;} configuration option, defaulting to <code>False</code>.</p>
    > <p>If set to <code>True</code>, pytest will attempt to identify modules that are part of <a href="https://packaging.python.org/en/latest/guides/packaging-namespace-packages">namespace packages</a> when importing modules.</p>
    > </li>
    > <li>
    > <p><a href="https://redirect.github.com/pytest-dev/pytest/issues/11653">#11653</a>: Added the new <code>verbosity_test_cases</code>{.interpreted-text role=&quot;confval&quot;} configuration option for fine-grained control of test execution verbosity.
    > See <code>Fine-grained verbosity &lt;pytest.fine_grained_verbosity&gt;</code>{.interpreted-text role=&quot;ref&quot;} for more details.</p>
    > </li>
    > </ul>
    > <h2>Improvements</h2>
    > <ul>
    > <li>
    > <p><a href="https://redirect.github.com/pytest-dev/pytest/issues/10865">#10865</a>: <code>pytest.warns</code>{.interpreted-text role=&quot;func&quot;} now validates that <code>warnings.warn</code>{.interpreted-text role=&quot;func&quot;} was called with a [str]{.title-ref} or a [Warning]{.title-ref}.
    > Currently in Python it is possible to use other types, however this causes an exception when <code>warnings.filterwarnings</code>{.interpreted-text role=&quot;func&quot;} is used to filter those warnings (see [CPython <a href="https://redirect.github.com/pytest-dev/pytest/issues/103577">#103577</a>](<a href="https://redirect.github.com/python/cpython/issues/103577">python/cpython#103577</a>) for a discussion).
    > While this can be considered a bug in CPython, we decided to put guards in pytest as the error message produced without this check in place is confusing.</p>
    > </li>
    > <li>
    > <p><a href="https://redirect.github.com/pytest-dev/pytest/issues/11311">#11311</a>: When using <code>--override-ini</code> for paths in invocations without a configuration file defined, the current working directory is used
    > as the relative directory.</p>
    > <p>Previoulsy this would raise an <code>AssertionError</code>{.interpreted-text role=&quot;class&quot;}.</p>
    > </li>
    > <li>
    > <p><a href="https://redirect.github.com/pytest-dev/pytest/issues/11475">#11475</a>: <code>--import-mode=importlib &lt;import-mode-importlib&gt;</code>{.interpreted-text role=&quot;ref&quot;} now tries to import modules using the standard import mechanism (but still without changing :py<code>sys.path</code>{.interpreted-text role=&quot;data&quot;}), falling back to importing modules directly only if that fails.</p>
    > <p>This means that installed packages will be imported under their canonical name if possible first, for example <code>app.core.models</code>, instead of having the module name always be derived from their path (for example <code>.env310.lib.site_packages.app.core.models</code>).</p>
    > </li>
    > <li>
    > <p><a href="https://redirect.github.com/pytest-dev/pytest/issues/11801">#11801</a>: Added the <code>iter_parents() &lt;_pytest.nodes.Node.iter_parents&gt;</code>{.interpreted-text role=&quot;func&quot;} helper method on nodes.
    > It is similar to <code>listchain &lt;_pytest.nodes.Node.listchain&gt;</code>{.interpreted-text role=&quot;func&quot;}, but goes from bottom to top, and returns an iterator, not a list.</p>
    > </li>
    > <li>
    > <p><a href="https://redirect.github.com/pytest-dev/pytest/issues/11850">#11850</a>: Added support for <code>sys.last_exc</code>{.interpreted-text role=&quot;data&quot;} for post-mortem debugging on Python&gt;=3.12.</p>
    > </li>
    > <li>
    > <p><a href="https://redirect.github.com/pytest-dev/pytest/issues/11962">#11962</a>: In case no other suitable candidates for configuration file are found, a <code>pyproject.toml</code> (even without a <code>[tool.pytest.ini_options]</code> table) will be considered as the configuration file and define the <code>rootdir</code>.</p>
    > </li>
    > <li>
    > <p><a href="https://redirect.github.com/pytest-dev/pytest/issues/11978">#11978</a>: Add <code>--log-file-mode</code> option to the logging plugin, enabling appending to log-files. This option accepts either <code>&quot;w&quot;</code> or <code>&quot;a&quot;</code> and defaults to <code>&quot;w&quot;</code>.</p>
    > <p>Previously, the mode was hard-coded to be <code>&quot;w&quot;</code> which truncates the file before logging.</p>
    > </li>
    > </ul>
    > <!-- raw HTML omitted -->
    > </blockquote>
    > <p>... (truncated)</p>
    > </details>
    > <details>
    > <summary>Commits</summary>
    > <ul>
    > <li><a href="https://github.com/pytest-dev/pytest/commit/81653ee385f4c62ee7e64502a7b7530096553115"><code>81653ee</code></a> Adjust changelog manually for 8.1.1</li>
    > <li><a href="https://github.com/pytest-dev/pytest/commit/e60b4b9ed80f761e3a51868a01338911a567b093"><code>e60b4b9</code></a> Prepare release version 8.1.1</li>
    > <li><a href="https://github.com/pytest-dev/pytest/commit/15fbe57c44fed6737f5c6dad99cf4437b6755a6c"><code>15fbe57</code></a> [8.1.x] Revert legacy path removals (<a href="https://redirect.github.com/pytest-dev/pytest/issues/12093">#12093</a>)</li>
    > <li><a href="https://github.com/pytest-dev/pytest/commit/86c3aab005a98de7e12ee5e37782837f5db70ac3"><code>86c3aab</code></a> [8.1.x] Do not import duplicated modules with --importmode=importlib (<a href="https://redirect.github.com/pytest-dev/pytest/issues/12077">#12077</a>)</li>
    > <li><a href="https://github.com/pytest-dev/pytest/commit/5b82b0cd20c3adcc21f34ae30c595c7355a87e23"><code>5b82b0c</code></a> [8.1.x] Yank version 8.1.0 (<a href="https://redirect.github.com/pytest-dev/pytest/issues/12076">#12076</a>)</li>
    > <li><a href="https://github.com/pytest-dev/pytest/commit/0a536810dc5f51dac99bdb90dde06704b5aa034e"><code>0a53681</code></a> Merge pull request <a href="https://redirect.github.com/pytest-dev/pytest/issues/12054">#12054</a> from pytest-dev/release-8.1.0</li>
    > <li><a href="https://github.com/pytest-dev/pytest/commit/b9a167f9bbbd6eda4f0360c5bf5b7f5af50f2bc4"><code>b9a167f</code></a> Prepare release version 8.1.0</li>
    > <li><a href="https://github.com/pytest-dev/pytest/commit/00043f7f1047b29fdaeb18e169fe9d6146988cb8"><code>00043f7</code></a> Merge pull request <a href="https://redirect.github.com/pytest-dev/pytest/issues/12038">#12038</a> from bluetech/fixtures-rm-arg2index</li>
    > <li><a href="https://github.com/pytest-dev/pytest/commit/f4e10251a4a003495b5228cea421d4de5fa0ce89"><code>f4e1025</code></a> Merge pull request <a href="https://redirect.github.com/pytest-dev/pytest/issues/12048">#12048</a> from bluetech/fixture-teardown-excgroup</li>
    > <li><a href="https://github.com/pytest-dev/pytest/commit/43492f5707b38dab9b62dfb829bb41a13579629f"><code>43492f5</code></a> Merge pull request <a href="https://redirect.github.com/pytest-dev/pytest/issues/12051">#12051</a> from jakkdl/test_debugging_pythonbreakpoint</li>
    > <li>Additional commits viewable in <a href="https://github.com/pytest-dev/pytest/compare/1.0.0b3...8.1.1">compare view</a></li>
    > </ul>
    > </details>
    > <br />
    > 
    > 
    > Dependabot will resolve any conflicts with this PR as long as you don't alter it yourself. You can also trigger a rebase manually by commenting `@dependabot rebase`.
    > 
    > [//]: # (dependabot-automerge-start)
    > [//]: # (dependabot-automerge-end)
    > 
    > ---
    > 
    > <details>
    > <summary>Dependabot commands and options</summary>
    > <br />
    > 
    > You can trigger Dependabot actions by commenting on this PR:
    > - `@dependabot rebase` will rebase this PR
    > - `@dependabot recreate` will recreate this PR, overwriting any edits that have been made to it
    > - `@dependabot merge` will merge this PR after your CI passes on it
    > - `@dependabot squash and merge` will squash and merge this PR after your CI passes on it
    > - `@dependabot cancel merge` will cancel a previously requested merge and block automerging
    > - `@dependabot reopen` will reopen this PR if it is closed
    > - `@dependabot close` will close this PR and stop Dependabot recreating it. You can achieve the same result by closing it manually
    > - `@dependabot show <dependency name> ignore conditions` will show all of the ignore conditions of the specified dependency
    > - `@dependabot ignore this major version` will close this PR and stop Dependabot creating any more for this major version (unless you reopen the PR or upgrade to it yourself)
    > - `@dependabot ignore this minor version` will close this PR and stop Dependabot creating any more for this minor version (unless you reopen the PR or upgrade to it yourself)
    > - `@dependabot ignore this dependency` will close this PR and stop Dependabot creating any more for this dependency (unless you reopen the PR or upgrade to it yourself)
    > 
    > 
    > </details>
- [dependabot[bot]](https://github.com/Living-with-machines/MapReader/pull/382)
    > Updates the requirements on [torchvision](https://github.com/pytorch/vision) to permit the latest version.
    > <details>
    > <summary>Release notes</summary>
    > <p><em>Sourced from <a href="https://github.com/pytorch/vision/releases">torchvision's releases</a>.</em></p>
    > <blockquote>
    > <h2>TorchVision 0.17.2 Release</h2>
    > <p>This is a patch release, which is compatible with <a href="https://github.com/pytorch/pytorch/releases/tag/v2.2.2">PyTorch 2.2.2</a>. There are no new features added.</p>
    > </blockquote>
    > </details>
    > <details>
    > <summary>Commits</summary>
    > <ul>
    > <li><a href="https://github.com/pytorch/vision/commit/c1d70fe1aa3f37ecdc809311f6c238df900dfd19"><code>c1d70fe</code></a> Migrate the macOS runners label from macos-m1-12 to macos-m1-stable (<a href="https://redirect.github.com/pytorch/vision/issues/8346">#8346</a>)</li>
    > <li><a href="https://github.com/pytorch/vision/commit/8e64d60b4428d1ded1702fced5e4e0ce6e67da10"><code>8e64d60</code></a> Bump version to 0.17.2</li>
    > <li><a href="https://github.com/pytorch/vision/commit/4fd856bfbcf59a4da3a91f0e12515c7ef0709777"><code>4fd856b</code></a> Bump version for release 0.17.1 (<a href="https://redirect.github.com/pytorch/vision/issues/8271">#8271</a>)</li>
    > <li><a href="https://github.com/pytorch/vision/commit/20610ed3f8d79c8d1b414ffd6f317951b43dabfb"><code>20610ed</code></a> [Cherry-pick for 0.17.1] Fix convert_bounding_box_format when passing strings...</li>
    > <li><a href="https://github.com/pytorch/vision/commit/a0e8e6cfb1ee597f9c745f43c4ccf9f42d34c870"><code>a0e8e6c</code></a> [Cherry-pick for 0.17.1] add gdown as optional requirement for dataset GDrive...</li>
    > <li><a href="https://github.com/pytorch/vision/commit/b2383d44751bf85e58cfb9223bbf4e5961c09fa1"><code>b2383d4</code></a> [RELEASE-ONLY CHANGES] [Cherry-pick for 0.17] CI fix - Use pytest&lt;8 in unitte...</li>
    > <li><a href="https://github.com/pytorch/vision/commit/273182ae259c90e560bd7542f67df065d0fa735a"><code>273182a</code></a> empty</li>
    > <li><a href="https://github.com/pytorch/vision/commit/9c7fb73ff9e8d6882734ad5cb2012884cb09c152"><code>9c7fb73</code></a> Remove AWS credentials on workflows (<a href="https://redirect.github.com/pytorch/vision/issues/8207">#8207</a>)</li>
    > <li><a href="https://github.com/pytorch/vision/commit/49880a9fa0ac0ccaa37d8582dfa5e6237a46a4f7"><code>49880a9</code></a> [Cherry-pick for 0.17] Fix root path expansion in datasets.Kitti (<a href="https://redirect.github.com/pytorch/vision/issues/8165">#8165</a>)</li>
    > <li><a href="https://github.com/pytorch/vision/commit/2b193ecb3d8b303e1644f14afff587fb0970dd74"><code>2b193ec</code></a> [Cherry-pick for 0.17]  Fix TestElastic::test_transform on M1 (<a href="https://redirect.github.com/pytorch/vision/issues/8160">#8160</a>)</li>
    > <li>Additional commits viewable in <a href="https://github.com/pytorch/vision/compare/v0.11.1...v0.17.2">compare view</a></li>
    > </ul>
    > </details>
    > <br />
    > 
    > 
    > Dependabot will resolve any conflicts with this PR as long as you don't alter it yourself. You can also trigger a rebase manually by commenting `@dependabot rebase`.
    > 
    > [//]: # (dependabot-automerge-start)
    > [//]: # (dependabot-automerge-end)
    > 
    > ---
    > 
    > <details>
    > <summary>Dependabot commands and options</summary>
    > <br />
    > 
    > You can trigger Dependabot actions by commenting on this PR:
    > - `@dependabot rebase` will rebase this PR
    > - `@dependabot recreate` will recreate this PR, overwriting any edits that have been made to it
    > - `@dependabot merge` will merge this PR after your CI passes on it
    > - `@dependabot squash and merge` will squash and merge this PR after your CI passes on it
    > - `@dependabot cancel merge` will cancel a previously requested merge and block automerging
    > - `@dependabot reopen` will reopen this PR if it is closed
    > - `@dependabot close` will close this PR and stop Dependabot recreating it. You can achieve the same result by closing it manually
    > - `@dependabot show <dependency name> ignore conditions` will show all of the ignore conditions of the specified dependency
    > - `@dependabot ignore this major version` will close this PR and stop Dependabot creating any more for this major version (unless you reopen the PR or upgrade to it yourself)
    > - `@dependabot ignore this minor version` will close this PR and stop Dependabot creating any more for this minor version (unless you reopen the PR or upgrade to it yourself)
    > - `@dependabot ignore this dependency` will close this PR and stop Dependabot creating any more for this dependency (unless you reopen the PR or upgrade to it yourself)
    > 
    > 
    > </details>
- [dependabot[bot]](https://github.com/Living-with-machines/MapReader/pull/381)
    > Updates the requirements on [black](https://github.com/psf/black) to permit the latest version.
    > <details>
    > <summary>Release notes</summary>
    > <p><em>Sourced from <a href="https://github.com/psf/black/releases">black's releases</a>.</em></p>
    > <blockquote>
    > <h2>24.3.0</h2>
    > <h3>Highlights</h3>
    > <p>This release is a milestone: it fixes Black's first CVE security vulnerability. If you
    > run Black on untrusted input, or if you habitually put thousands of leading tab
    > characters in your docstrings, you are strongly encouraged to upgrade immediately to fix
    > <a href="https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2024-21503">CVE-2024-21503</a>.</p>
    > <p>This release also fixes a bug in Black's AST safety check that allowed Black to make
    > incorrect changes to certain f-strings that are valid in Python 3.12 and higher.</p>
    > <h3>Stable style</h3>
    > <ul>
    > <li>Don't move comments along with delimiters, which could cause crashes (<a href="https://redirect.github.com/psf/black/issues/4248">#4248</a>)</li>
    > <li>Strengthen AST safety check to catch more unsafe changes to strings. Previous versions
    > of Black would incorrectly format the contents of certain unusual f-strings containing
    > nested strings with the same quote type. Now, Black will crash on such strings until
    > support for the new f-string syntax is implemented. (<a href="https://redirect.github.com/psf/black/issues/4270">#4270</a>)</li>
    > <li>Fix a bug where line-ranges exceeding the last code line would not work as expected
    > (<a href="https://redirect.github.com/psf/black/issues/4273">#4273</a>)</li>
    > </ul>
    > <h3>Performance</h3>
    > <ul>
    > <li>Fix catastrophic performance on docstrings that contain large numbers of leading tab
    > characters. This fixes
    > <a href="https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2024-21503">CVE-2024-21503</a>.
    > (<a href="https://redirect.github.com/psf/black/issues/4278">#4278</a>)</li>
    > </ul>
    > <h3>Documentation</h3>
    > <ul>
    > <li>Note what happens when <code>--check</code> is used with <code>--quiet</code> (<a href="https://redirect.github.com/psf/black/issues/4236">#4236</a>)</li>
    > </ul>
    > </blockquote>
    > </details>
    > <details>
    > <summary>Changelog</summary>
    > <p><em>Sourced from <a href="https://github.com/psf/black/blob/main/CHANGES.md">black's changelog</a>.</em></p>
    > <blockquote>
    > <h2>24.3.0</h2>
    > <h3>Highlights</h3>
    > <p>This release is a milestone: it fixes Black's first CVE security vulnerability. If you
    > run Black on untrusted input, or if you habitually put thousands of leading tab
    > characters in your docstrings, you are strongly encouraged to upgrade immediately to fix
    > <a href="https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2024-21503">CVE-2024-21503</a>.</p>
    > <p>This release also fixes a bug in Black's AST safety check that allowed Black to make
    > incorrect changes to certain f-strings that are valid in Python 3.12 and higher.</p>
    > <h3>Stable style</h3>
    > <ul>
    > <li>Don't move comments along with delimiters, which could cause crashes (<a href="https://redirect.github.com/psf/black/issues/4248">#4248</a>)</li>
    > <li>Strengthen AST safety check to catch more unsafe changes to strings. Previous versions
    > of Black would incorrectly format the contents of certain unusual f-strings containing
    > nested strings with the same quote type. Now, Black will crash on such strings until
    > support for the new f-string syntax is implemented. (<a href="https://redirect.github.com/psf/black/issues/4270">#4270</a>)</li>
    > <li>Fix a bug where line-ranges exceeding the last code line would not work as expected
    > (<a href="https://redirect.github.com/psf/black/issues/4273">#4273</a>)</li>
    > </ul>
    > <h3>Performance</h3>
    > <ul>
    > <li>Fix catastrophic performance on docstrings that contain large numbers of leading tab
    > characters. This fixes
    > <a href="https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2024-21503">CVE-2024-21503</a>.
    > (<a href="https://redirect.github.com/psf/black/issues/4278">#4278</a>)</li>
    > </ul>
    > <h3>Documentation</h3>
    > <ul>
    > <li>Note what happens when <code>--check</code> is used with <code>--quiet</code> (<a href="https://redirect.github.com/psf/black/issues/4236">#4236</a>)</li>
    > </ul>
    > <h2>24.2.0</h2>
    > <h3>Stable style</h3>
    > <ul>
    > <li>Fixed a bug where comments where mistakenly removed along with redundant parentheses
    > (<a href="https://redirect.github.com/psf/black/issues/4218">#4218</a>)</li>
    > </ul>
    > <h3>Preview style</h3>
    > <ul>
    > <li>Move the <code>hug_parens_with_braces_and_square_brackets</code> feature to the unstable style
    > due to an outstanding crash and proposed formatting tweaks (<a href="https://redirect.github.com/psf/black/issues/4198">#4198</a>)</li>
    > <li>Fixed a bug where base expressions caused inconsistent formatting of ** in tenary
    > expression (<a href="https://redirect.github.com/psf/black/issues/4154">#4154</a>)</li>
    > <li>Checking for newline before adding one on docstring that is almost at the line limit
    > (<a href="https://redirect.github.com/psf/black/issues/4185">#4185</a>)</li>
    > <li>Remove redundant parentheses in <code>case</code> statement <code>if</code> guards (<a href="https://redirect.github.com/psf/black/issues/4214">#4214</a>).</li>
    > </ul>
    > <!-- raw HTML omitted -->
    > </blockquote>
    > <p>... (truncated)</p>
    > </details>
    > <details>
    > <summary>Commits</summary>
    > <ul>
    > <li><a href="https://github.com/psf/black/commit/552baf822992936134cbd31a38f69c8cfe7c0f05"><code>552baf8</code></a> Prepare release 24.3.0 (<a href="https://redirect.github.com/psf/black/issues/4279">#4279</a>)</li>
    > <li><a href="https://github.com/psf/black/commit/f00093672628d212b8965a8993cee8bedf5fe9b8"><code>f000936</code></a> Fix catastrophic performance in lines_with_leading_tabs_expanded() (<a href="https://redirect.github.com/psf/black/issues/4278">#4278</a>)</li>
    > <li><a href="https://github.com/psf/black/commit/7b5a657285f38126bf28483478bbd9ea928077ec"><code>7b5a657</code></a> Fix --line-ranges behavior when ranges are at EOF (<a href="https://redirect.github.com/psf/black/issues/4273">#4273</a>)</li>
    > <li><a href="https://github.com/psf/black/commit/1abcffc81816257985678f08c61584ed4287f22a"><code>1abcffc</code></a> Use regex where we ignore case on windows (<a href="https://redirect.github.com/psf/black/issues/4252">#4252</a>)</li>
    > <li><a href="https://github.com/psf/black/commit/719e67462c80574c81a96faa144886de6da84489"><code>719e674</code></a> Fix 4227: Improve documentation for --quiet --check (<a href="https://redirect.github.com/psf/black/issues/4236">#4236</a>)</li>
    > <li><a href="https://github.com/psf/black/commit/e5510afc06cd238cd0cba4095283943a870a7e7b"><code>e5510af</code></a> update plugin url for Thonny (<a href="https://redirect.github.com/psf/black/issues/4259">#4259</a>)</li>
    > <li><a href="https://github.com/psf/black/commit/6af7d1109693c4ad3af08ecbc34649c232b47a6d"><code>6af7d11</code></a> Fix AST safety check false negative (<a href="https://redirect.github.com/psf/black/issues/4270">#4270</a>)</li>
    > <li><a href="https://github.com/psf/black/commit/f03ee113c9f3dfeb477f2d4247bfb7de2e5f465c"><code>f03ee11</code></a> Ensure <code>blib2to3.pygram</code> is initialized before use (<a href="https://redirect.github.com/psf/black/issues/4224">#4224</a>)</li>
    > <li><a href="https://github.com/psf/black/commit/e4bfedbec2e8b10cc6b7b31442478f05db0ce06d"><code>e4bfedb</code></a> fix: Don't move comments while splitting delimiters (<a href="https://redirect.github.com/psf/black/issues/4248">#4248</a>)</li>
    > <li><a href="https://github.com/psf/black/commit/d0287e1f7558d97e6c0ebd6dc5bcb5b970e2bf8c"><code>d0287e1</code></a> Make trailing comma logic more concise (<a href="https://redirect.github.com/psf/black/issues/4202">#4202</a>)</li>
    > <li>Additional commits viewable in <a href="https://github.com/psf/black/compare/23.7.0...24.3.0">compare view</a></li>
    > </ul>
    > </details>
    > <br />
    > 
    > 
    > Dependabot will resolve any conflicts with this PR as long as you don't alter it yourself. You can also trigger a rebase manually by commenting `@dependabot rebase`.
    > 
    > [//]: # (dependabot-automerge-start)
    > [//]: # (dependabot-automerge-end)
    > 
    > ---
    > 
    > <details>
    > <summary>Dependabot commands and options</summary>
    > <br />
    > 
    > You can trigger Dependabot actions by commenting on this PR:
    > - `@dependabot rebase` will rebase this PR
    > - `@dependabot recreate` will recreate this PR, overwriting any edits that have been made to it
    > - `@dependabot merge` will merge this PR after your CI passes on it
    > - `@dependabot squash and merge` will squash and merge this PR after your CI passes on it
    > - `@dependabot cancel merge` will cancel a previously requested merge and block automerging
    > - `@dependabot reopen` will reopen this PR if it is closed
    > - `@dependabot close` will close this PR and stop Dependabot recreating it. You can achieve the same result by closing it manually
    > - `@dependabot show <dependency name> ignore conditions` will show all of the ignore conditions of the specified dependency
    > - `@dependabot ignore this major version` will close this PR and stop Dependabot creating any more for this major version (unless you reopen the PR or upgrade to it yourself)
    > - `@dependabot ignore this minor version` will close this PR and stop Dependabot creating any more for this minor version (unless you reopen the PR or upgrade to it yourself)
    > - `@dependabot ignore this dependency` will close this PR and stop Dependabot creating any more for this dependency (unless you reopen the PR or upgrade to it yourself)
    > 
    > 
    > </details>
- [dependabot[bot]](https://github.com/Living-with-machines/MapReader/pull/380)
    > Updates the requirements on [pytest-cov](https://github.com/pytest-dev/pytest-cov) to permit the latest version.
    > <details>
    > <summary>Changelog</summary>
    > <p><em>Sourced from <a href="https://github.com/pytest-dev/pytest-cov/blob/master/CHANGELOG.rst">pytest-cov's changelog</a>.</em></p>
    > <blockquote>
    > <h2>5.0.0 (2024-03-24)</h2>
    > <ul>
    > <li>Removed support for xdist rsync (now deprecated).
    > Contributed by Matthias Reichenbach in <code>[#623](https://github.com/pytest-dev/pytest-cov/issues/623) &lt;https://github.com/pytest-dev/pytest-cov/pull/623&gt;</code>_.</li>
    > <li>Switched docs theme to Furo.</li>
    > <li>Various legacy Python cleanup and CI improvements.
    > Contributed by Christian Clauss and Hugo van Kemenade in
    > <code>[#630](https://github.com/pytest-dev/pytest-cov/issues/630) &lt;https://github.com/pytest-dev/pytest-cov/pull/630&gt;</code><em>,
    > <code>[#631](https://github.com/pytest-dev/pytest-cov/issues/631) &lt;https://github.com/pytest-dev/pytest-cov/pull/631&gt;</code></em>,
    > <code>[#632](https://github.com/pytest-dev/pytest-cov/issues/632) &lt;https://github.com/pytest-dev/pytest-cov/pull/632&gt;</code>_ and
    > <code>[#633](https://github.com/pytest-dev/pytest-cov/issues/633) &lt;https://github.com/pytest-dev/pytest-cov/pull/633&gt;</code>_.</li>
    > <li>Added a <code>pyproject.toml</code> example in the docs.
    > Contributed by Dawn James in <code>[#626](https://github.com/pytest-dev/pytest-cov/issues/626) &lt;https://github.com/pytest-dev/pytest-cov/pull/626&gt;</code>_.</li>
    > <li>Modernized project's pre-commit hooks to use ruff. Initial POC contributed by
    > Christian Clauss in <code>[#584](https://github.com/pytest-dev/pytest-cov/issues/584) &lt;https://github.com/pytest-dev/pytest-cov/pull/584&gt;</code>_.</li>
    > </ul>
    > <h2>4.1.0 (2023-05-24)</h2>
    > <ul>
    > <li>Updated CI with new Pythons and dependencies.</li>
    > <li>Removed rsyncdir support. This makes pytest-cov compatible with xdist 3.0.
    > Contributed by Sorin Sbarnea in <code>[#558](https://github.com/pytest-dev/pytest-cov/issues/558) &lt;https://github.com/pytest-dev/pytest-cov/pull/558&gt;</code>_.</li>
    > <li>Optimized summary generation to not be performed if no reporting is active (for example,
    > when <code>--cov-report=''</code> is used without <code>--cov-fail-under</code>).
    > Contributed by Jonathan Stewmon in <code>[#589](https://github.com/pytest-dev/pytest-cov/issues/589) &lt;https://github.com/pytest-dev/pytest-cov/pull/589&gt;</code>_.</li>
    > <li>Added support for JSON reporting.
    > Contributed by Matthew Gamble in <code>[#582](https://github.com/pytest-dev/pytest-cov/issues/582) &lt;https://github.com/pytest-dev/pytest-cov/pull/582&gt;</code>_.</li>
    > <li>Refactored code to use f-strings.
    > Contributed by Mark Mayo in <code>[#572](https://github.com/pytest-dev/pytest-cov/issues/572) &lt;https://github.com/pytest-dev/pytest-cov/pull/572&gt;</code>_.</li>
    > <li>Fixed a skip in the test suite for some old xdist.
    > Contributed by a bunch of people in <code>[#565](https://github.com/pytest-dev/pytest-cov/issues/565) &lt;https://github.com/pytest-dev/pytest-cov/pull/565&gt;</code>_.</li>
    > </ul>
    > <h2>4.0.0 (2022-09-28)</h2>
    > <p><strong>Note that this release drops support for multiprocessing.</strong></p>
    > <ul>
    > <li>
    > <p><code>--cov-fail-under</code> no longer causes <code>pytest --collect-only</code> to fail
    > Contributed by Zac Hatfield-Dodds in <code>[#511](https://github.com/pytest-dev/pytest-cov/issues/511) &lt;https://github.com/pytest-dev/pytest-cov/pull/511&gt;</code>_.</p>
    > </li>
    > <li>
    > <p>Dropped support for multiprocessing (mostly because <code>issue 82408 &lt;https://github.com/python/cpython/issues/82408&gt;</code>_). This feature was
    > mostly working but very broken in certain scenarios and made the test suite very flaky and slow.</p>
    > <p>There is builtin multiprocessing support in coverage and you can migrate to that. All you need is this in your
    > <code>.coveragerc</code>::</p>
    > <p>[run]
    > concurrency = multiprocessing</p>
    > </li>
    > </ul>
    > <!-- raw HTML omitted -->
    > </blockquote>
    > <p>... (truncated)</p>
    > </details>
    > <details>
    > <summary>Commits</summary>
    > <ul>
    > <li><a href="https://github.com/pytest-dev/pytest-cov/commit/5295ce01c84262cec88f31255e9ac538718f3047"><code>5295ce0</code></a> Bump version: 4.1.0 → 5.0.0</li>
    > <li><a href="https://github.com/pytest-dev/pytest-cov/commit/1181b067972bf94569f8011f3b18f271690f9ab1"><code>1181b06</code></a> Update changelog.</li>
    > <li><a href="https://github.com/pytest-dev/pytest-cov/commit/9757222e2e044361e70125ebdd96e5eb87395983"><code>9757222</code></a> Fix a minor grammar error (<a href="https://redirect.github.com/pytest-dev/pytest-cov/issues/636">#636</a>)</li>
    > <li><a href="https://github.com/pytest-dev/pytest-cov/commit/9f5cd81a0dbe3fe41681efdbef516c08988fe8ff"><code>9f5cd81</code></a> Cleanup releasing instructions. Closes <a href="https://redirect.github.com/pytest-dev/pytest-cov/issues/616">#616</a>.</li>
    > <li><a href="https://github.com/pytest-dev/pytest-cov/commit/93b5047ec5050d63c10a6fe16a09b671a7a03df8"><code>93b5047</code></a> Add test for pyproject.toml loading without explicit --cov-config. Ref <a href="https://redirect.github.com/pytest-dev/pytest-cov/issues/508">#508</a>.</li>
    > <li><a href="https://github.com/pytest-dev/pytest-cov/commit/ff50860d7c67b920503745d92a3f0944cf41f982"><code>ff50860</code></a> docs: add config instructions for pyproject.toml.</li>
    > <li><a href="https://github.com/pytest-dev/pytest-cov/commit/4a5a4b5fa4b1c63ddcab5cbc1813798c9b6f1d36"><code>4a5a4b5</code></a> Keep GitHub Actions up to date with GitHub's Dependabot</li>
    > <li><a href="https://github.com/pytest-dev/pytest-cov/commit/1d7f55963d5138f41c452a946f7cca7e0b6ee8b2"><code>1d7f559</code></a> Fix or remove URLs that are causing docs tests to fail</li>
    > <li><a href="https://github.com/pytest-dev/pytest-cov/commit/6a5af8e85b8242ac815f33e26adf9068f5f0ebc3"><code>6a5af8e</code></a> Update changelog.</li>
    > <li><a href="https://github.com/pytest-dev/pytest-cov/commit/d9fe8dfed15023d3410dd299c5092e755b8981c2"><code>d9fe8df</code></a> Switch to furo. Closes <a href="https://redirect.github.com/pytest-dev/pytest-cov/issues/618">#618</a>.</li>
    > <li>Additional commits viewable in <a href="https://github.com/pytest-dev/pytest-cov/compare/v4.1.0...v5.0.0">compare view</a></li>
    > </ul>
    > </details>
    > <br />
    > 
    > 
    > Dependabot will resolve any conflicts with this PR as long as you don't alter it yourself. You can also trigger a rebase manually by commenting `@dependabot rebase`.
    > 
    > [//]: # (dependabot-automerge-start)
    > [//]: # (dependabot-automerge-end)
    > 
    > ---
    > 
    > <details>
    > <summary>Dependabot commands and options</summary>
    > <br />
    > 
    > You can trigger Dependabot actions by commenting on this PR:
    > - `@dependabot rebase` will rebase this PR
    > - `@dependabot recreate` will recreate this PR, overwriting any edits that have been made to it
    > - `@dependabot merge` will merge this PR after your CI passes on it
    > - `@dependabot squash and merge` will squash and merge this PR after your CI passes on it
    > - `@dependabot cancel merge` will cancel a previously requested merge and block automerging
    > - `@dependabot reopen` will reopen this PR if it is closed
    > - `@dependabot close` will close this PR and stop Dependabot recreating it. You can achieve the same result by closing it manually
    > - `@dependabot show <dependency name> ignore conditions` will show all of the ignore conditions of the specified dependency
    > - `@dependabot ignore this major version` will close this PR and stop Dependabot creating any more for this major version (unless you reopen the PR or upgrade to it yourself)
    > - `@dependabot ignore this minor version` will close this PR and stop Dependabot creating any more for this minor version (unless you reopen the PR or upgrade to it yourself)
    > - `@dependabot ignore this dependency` will close this PR and stop Dependabot creating any more for this dependency (unless you reopen the PR or upgrade to it yourself)
    > 
    > 
    > </details>
- [dependabot[bot]](https://github.com/Living-with-machines/MapReader/pull/383)
    > Updates the requirements on [flake8](https://github.com/pycqa/flake8) to permit the latest version.
    > <details>
    > <summary>Commits</summary>
    > <ul>
    > <li><a href="https://github.com/PyCQA/flake8/commit/88a4f9b2f48fc44b025a48fa6a8ac7cc89ef70e0"><code>88a4f9b</code></a> Release 7.0.0</li>
    > <li><a href="https://github.com/PyCQA/flake8/commit/6f3a60dd460f473aa0017f3d6b3164106b0d2fdc"><code>6f3a60d</code></a> Merge pull request <a href="https://redirect.github.com/pycqa/flake8/issues/1906">#1906</a> from PyCQA/upgrade-pyflakes</li>
    > <li><a href="https://github.com/PyCQA/flake8/commit/cde8570df3bf4b647dfa65a97613fb325a9f1bbd"><code>cde8570</code></a> upgrade pyflakes to 3.2.x</li>
    > <li><a href="https://github.com/PyCQA/flake8/commit/2ab9d76639c43f7462191e46c18afb31a15e9e36"><code>2ab9d76</code></a> Merge pull request <a href="https://redirect.github.com/pycqa/flake8/issues/1903">#1903</a> from PyCQA/pre-commit-ci-update-config</li>
    > <li><a href="https://github.com/PyCQA/flake8/commit/e27611f1eadc16a5bd02125aa8a054c632d3b0c7"><code>e27611f</code></a> [pre-commit.ci] pre-commit autoupdate</li>
    > <li><a href="https://github.com/PyCQA/flake8/commit/9d20be1d1f07513b82088860caa7130e2ac49618"><code>9d20be1</code></a> Merge pull request <a href="https://redirect.github.com/pycqa/flake8/issues/1902">#1902</a> from PyCQA/pre-commit-ci-update-config</li>
    > <li><a href="https://github.com/PyCQA/flake8/commit/06c1503842ee90a4cca5ed57908c0f27595a6f4d"><code>06c1503</code></a> [pre-commit.ci] auto fixes from pre-commit.com hooks</li>
    > <li><a href="https://github.com/PyCQA/flake8/commit/b67ce03a4a9c9902fea163021a844f34287ee6bc"><code>b67ce03</code></a> Fix bugbear lints</li>
    > <li><a href="https://github.com/PyCQA/flake8/commit/c8801c129ab3138c4f3db4841d76bb30ed8e3f8c"><code>c8801c1</code></a> [pre-commit.ci] pre-commit autoupdate</li>
    > <li><a href="https://github.com/PyCQA/flake8/commit/045f297f89a4d6b7f1bb6dc6e62f6eb506aec320"><code>045f297</code></a> Merge pull request <a href="https://redirect.github.com/pycqa/flake8/issues/1893">#1893</a> from PyCQA/pre-commit-ci-update-config</li>
    > <li>Additional commits viewable in <a href="https://github.com/pycqa/flake8/compare/6.0.0...7.0.0">compare view</a></li>
    > </ul>
    > </details>
    > <br />
    > 
    > 
    > Dependabot will resolve any conflicts with this PR as long as you don't alter it yourself. You can also trigger a rebase manually by commenting `@dependabot rebase`.
    > 
    > [//]: # (dependabot-automerge-start)
    > [//]: # (dependabot-automerge-end)
    > 
    > ---
    > 
    > <details>
    > <summary>Dependabot commands and options</summary>
    > <br />
    > 
    > You can trigger Dependabot actions by commenting on this PR:
    > - `@dependabot rebase` will rebase this PR
    > - `@dependabot recreate` will recreate this PR, overwriting any edits that have been made to it
    > - `@dependabot merge` will merge this PR after your CI passes on it
    > - `@dependabot squash and merge` will squash and merge this PR after your CI passes on it
    > - `@dependabot cancel merge` will cancel a previously requested merge and block automerging
    > - `@dependabot reopen` will reopen this PR if it is closed
    > - `@dependabot close` will close this PR and stop Dependabot recreating it. You can achieve the same result by closing it manually
    > - `@dependabot show <dependency name> ignore conditions` will show all of the ignore conditions of the specified dependency
    > - `@dependabot ignore this major version` will close this PR and stop Dependabot creating any more for this major version (unless you reopen the PR or upgrade to it yourself)
    > - `@dependabot ignore this minor version` will close this PR and stop Dependabot creating any more for this minor version (unless you reopen the PR or upgrade to it yourself)
    > - `@dependabot ignore this dependency` will close this PR and stop Dependabot creating any more for this dependency (unless you reopen the PR or upgrade to it yourself)
    > 
    > 
    > </details>

### Files

#### Added

- .github/dependabot.yml
- .github/workflows/dependapot-review.yml

#### Modified

- .github/workflows/mr_ci.yml
- .github/workflows/mr_pip_ci.yml
- .github/workflows/publish-to-pypi.yml
- .github/workflows/publish-to-test-pypi.yml
- .pre-commit-config.yaml
- .ruff.toml
- docs/source/Install.rst
- mapreader/annotate/utils.py
- mapreader/classify/classifier.py
- setup.py

## [v1.1.5](https://github.com/Living-with-machines/MapReader/releases/tag/v1.1.5) (2024-04-04)

### Summary

The changelog summarizes recent updates and fixes to the MapReader project as follows:

#### Key Changes:
- Users can now choose to download files in parallel or sequentially to manage rate limits effectively.
- Added the ability for users to specify file names in the downloader function.
- Removed outdated conda installation instructions from documentation.
- Links to the contribution guide were updated, and information about setting up the development environment and running tests was added.
- Minor fixes were made throughout the codebase, including multiple list corrections and adjustments to compatibility (e.g., Python version updates).
- Reintroduction of the `square_cuts` option for image patching to maintain backwards compatibility, with a deprecation warning for future reference.
- Enhancements to the reprojection of coordinates to ensure accuracy when converting between coordinate systems.
- The author list in the documentation was reordered.

#### Documentation Updates:
- Various markdown and reStructuredText files were updated, including the README, installation instructions, and event details.

Overall, the changes focus on improving user experience, enhancing codebase functionality, and ensuring better documentation and compatibility.

### Commit messages

- allow users to specify whether to download in parallel
- Update Events.rst    Data/Culture workshop details added
- fix lists
- fix lists
- more fix lists
- Update Events.rst
- community calls
- allow users to specify file names in downloader
- comment out conda install
- fix link to contribution guide
- add info about dev environment and tests
- add pyogrio to dependencies
- supress decompression bomb error
- readd square cuts option
- fix transform
- unsupress decompression bomb error
- add tests
- Update supported python versions
- Update setup.py
- Update setup.py - add cartopy
- add commas
- Update cartopy instructions
- Update author list
- Update Install.rst

### Pull requests

- [kmcdono2](https://github.com/Living-with-machines/MapReader/pull/362)
    > ### Summary
    > 
    > Describe the problem you're trying to fix in this pull request.
    > Please reference any related issue and use fixes/close to automatically close them, if pertinent. For example: "Fixes #58", or "Addresses (but does not close) #238". -->
    > 
    > Fixes #<NUM>
    > 
    > ### Describe your changes
    > 
    > Describe changes/updates you have made to the code.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [ ] Self-review code
    > - [ ] Ensure submission passes current tests
    > - [ ] Add tests
    > - [ ] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/363)
    > ### Summary
    > 
    > This allows users to specify whether to download in parallel vs sequentially. 
    > Sequentially is slower but can overcome problems to do with rate limits when downloading from a tileserver.
    > 
    > ### Describe your changes
    > 
    > Feed the download_in_parallel argument up to download methods.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/366)
    > ### Summary
    > 
    > This removes the conda install instructions from the docs since it is outdated. 
    > Addresses #365
    > Addresses #364 
    > 
    > We should try to fix install asap as per #162 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Ensure submission passes current tests
    > - [x] Update relevant docs
    > - [x] Check on readthedocs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/372)
    > ### Summary
    > 
    > This PR:
    > - updates the link to contribution guide in the README
    > - adds info about installing MapReader in development mode
    > - adds info about how to run MapReader tests
    > 
    > Fixes #371 
    > Addresses #369 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/373)
    > ### Summary
    > 
    > In v1.1.3 I removed the `square_cuts` argument when patchifying images. 
    > 
    > In v1.1.2 or less, if you set `square_cuts=True`, when you reached an edge of an image, if your remaining 'slice' was less than the patch size you'd specified, you would move back a bit to create a square patch with some overlap with the previous patch. This essentially ensured all patches were square.
    > In v1.1.3+, I removed this in favour of padding the patches at edges. This still means you end up with square patches but if your 'slice' was less than the patch size you pad with x no. of pixels to create a square patch.
    > 
    > This PR re-adds the square-cuts argument to ensure bakcwards compatibility but with a deprecation warning.
    > 
    > ### Describe your changes
    > 
    > Add `_patchify_by_pixel_square()` method for the square cuts. This is just copy-paste of previous code and gets called if you set `square_cuts=True`.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    > - [ ] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/374)
    > ### Summary
    > 
    > Thanks to @DGalexander for flagging this.
    > 
    > This PR fixes mapreader's reproject coords functions when converting between coordinate systems.
    > 
    > ### Describe your changes
    > 
    > - coords are transformed from bounds:  (left, bottom, right, top) == (xmin, ymin, xmax, ymax)
    > - always_xy is set to True to ensure no variation between crs ordering
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/377)
    > ### Summary
    > 
    > Addresses #364 
    > 
    > ### Describe your changes
    > 
    > - Update supported python versions in docs
    > - Update setup.py to be 3.8+
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Ensure submission passes current tests
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/378)
    > ### Summary
    > 
    > As per https://github.com/openjournals/joss-reviews/issues/6434 we are updating author list.
    > 
    > ### Describe your changes
    > 
    > Reorder authors
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Ensure submission passes current tests
    > - [x] Check paper on GH actions
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?

### Files

#### Modified

- README.md
- docs/source/Contribution-guide/Code.rst
- docs/source/Events.rst
- docs/source/Install.rst
- mapreader/download/downloader.py
- mapreader/download/sheet_downloader.py
- mapreader/load/geo_utils.py
- mapreader/load/images.py
- paper/paper.md
- setup.py
- tests/test_load/test_geo_utils.py
- tests/test_load/test_images.py

## [v1.1.4](https://github.com/Living-with-machines/MapReader/releases/tag/v1.1.4) (2024-02-23)

### Summary

The changelog reflects a series of updates and fixes to the MapReader project, primarily focusing on worked examples and documentation. Key changes include:

- **Commits**: Enhanced the worked examples and fixed bugs related to the annotator and plants examples. Various notebooks, including those related to geospatial data and MNIST classification, have been updated along with improvements to project documentation.

- **Pull Requests**: PR [rwood-97](https://github.com/Living-with-machines/MapReader/pull/357) addresses issues stemming from outdated worked examples that were not compatible with MapReader v1.1.4. It includes fixes to the annotator and optimizes the output by reducing unnecessary columns.

- **Files**: New files were added, including a README for workshop notebooks and geospatial annotations. Several files have been modified, including FAQs and various notebooks for different classification tasks, while outdated notebooks have been removed.

Overall, the updates are aimed at improving functionality, user experience, and documentation across the project.

### Commit messages

- update annotate worked examples
- allow for patches with no parent
- fix plants worked example
- update geospatial pipeline
- update context notebook
- add readme for workshop notebooks
- update mnist notebook
- update docs
- Update Worked-examples.rst

### Pull requests

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/357)
    > ### Summary
    > 
    > Our worked examples were all slightly outdated and so some users were having trouble when running them. 
    > This PR updates them to work with MapReader v1.1.4.
    > 
    > Fixes #355 
    > 
    > ### Describe your changes
    > 
    > - Fix annotator for if you don't have urls
    > - Only return important columns
    > - Only add to parents if parents is not None
    > 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?

### Files

#### Added

- worked_examples/geospatial/context_classification_one_inch_maps/annotations_one_inch/rail_space_#rw#.csv
- worked_examples/geospatial/workshop_june_2023/README.md

#### Modified

- docs/source/Contribution-guide/Contribution-guide.rst
- docs/source/Developers-guide.rst
- docs/source/Project-cv.rst
- docs/source/Worked-examples/Worked-examples.rst
- mapreader/annotate/annotator.py
- mapreader/load/images.py
- worked_examples/annotation/how-to-annotate-model-predictions.ipynb
- worked_examples/annotation/how-to-annotate-patches.ipynb
- worked_examples/geospatial/classification_one_inch_maps/Pipeline.ipynb
- worked_examples/geospatial/classification_one_inch_maps/annotations_one_inch/rail_space_#rw#.csv
- worked_examples/geospatial/context_classification_one_inch_maps/Pipeline.ipynb
- worked_examples/non-geospatial/classification_mnist/Pipeline.ipynb
- worked_examples/non-geospatial/classification_mnist/annotations_mnist/mnist_#kasra#.csv
- worked_examples/non-geospatial/classification_plant_phenotype/Pipeline.ipynb

#### Removed

- docs/source/Worked-examples/mnist_pipeline.ipynb
- docs/source/Worked-examples/one_inch_pipeline.ipynb
- docs/source/Worked-examples/plant_pipeline.ipynb

## [v1.1.3](https://github.com/Living-with-machines/MapReader/releases/tag/v1.1.3) (2024-02-21)

### Summary

The changelog details updates and changes made to the MapReader project, characterized by significant enhancements in model prediction annotation, contextual data handling, and code clarity. Key modifications include:

1. **Feature Additions**:
   - Annotations loader can now create a patch context dataset, enabling users to annotate at the context level.
   - Introduction of a post-processing script for filtering out false positives relative to predicted labels.
   - Added a `filter_for` argument in the annotator class to facilitate filtering and improving annotation accuracy.

2. **Code Improvements**:
   - Updated and clarified naming conventions in several model and dataset components for better understanding.
   - Simplified the architecture by removing redundant classifiers and combining context functionalities into a unified model framework.
   - Fixed numerous bugs related to annotation processing and image handling, particularly concerning how edge patches are processed.

3. **Documentation and Testing**:
   - Enhanced documentation throughout the project to support new features and updates.
   - Significant focus on ensuring backward compatibility and the addition of new tests to validate the changes made.

4. **File Changes**:
   - New files were added for post-processing scripts, test cases, and example notebooks.
   - Several existing files were modified to improve code structure and readability.
   - Redundant files were removed, streamlining the project.

Overall, these changes enhance the usability and functionality of the MapReader tool for better model prediction annotation and contextual understanding in data processing.

### Commit messages

- enable annotations loader to create patch context dataset
- align classifier_context to classifier
- update trainable_col arg name
- fix color printing
- always return images as tuple
- process inputs as a tuple
- update attribute names in custom model for clarity
- update confusing language in params2optimize
- add context option for generate_layerwise_lrs
- remove classifier context (now all in one)
- remove context container from init imports
- add docs on how to use context model
- add filter_for arg to annotator
- Merge branch 'kallewesterling/issue166' into analyse_preds
- Merge branch 'kallewesterling/issue166' into analyse_preds
- add printing of filter
- add filter_for to docs
- add notebook for how to annotate model predictions
- enable easier saving of predictions to csv
- add post processing script
- add docstrings, allow user to specify conf
- skip edge patches, allow new labels
- replace `square_cuts` with padding at edge patches
- return df after eval
- update context saving
- remove square_cuts arg from tests
- ensure geotiffs are saved correctly
- fix context for annotator
- allow users to annotate at context-level
- use iloc not at for getting data
- fix load annotations
- rename context dataset trasnforms for clarity
- only add context annotations to annotated patches
- keep all cols when saving
- return only context image for context datasets
- remove context annotations from annotator
- force image_id index
- use total_df to build context images
- force image_id index
- add tests
- Add post-processing docs
- add suggestion
- Update codecov fail in CI
- add tests
- update docs
- fix typo
- fix index map vs apply
- update sample annots file
- update/fix tests
- update subtitle
- Update affiliations
- Fix (?) references
- fix typo
- fix filter for
- update notebook
- add worked example for context classification
- update/add tests
- fix test_annotations_loader
- update test_classifier
- ensure pixel stats are correct for edge patches
- add tests for geotiff saving (edge patches)
- add tests for datasets
- add datasets tests and fix parhugin code
- update docs
- only save important cols in annotator
- ensure backwards compatibility
- test backward compatibility
- better test for scramble frame

### Pull requests

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/342)
    > ### Summary
    > 
    > As per #339, this PR implements a post-processing script so that users can filter out false positives. 
    > This works for linear features or anything where you expect multiple patches to be clustered but solo patches would be false positive.
    > It also adds a `save_predictions()` method the classifier to make sure predictions and confidence scores are saved in format expected for post-processing.
    > 
    > Fixes #218 
    > Addresses #339 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Allow users to pick lowest conf for which to change label
    > - [x] Check for edge cases - overlapping patches and non-square patches
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    > - [ ] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [x] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/319)
    > ### Summary
    > 
    > Since we have run the 2nd ed model on a set of 1st ed patches, we need a method of looking at the predicted labels and marking them as correct/incorrect. It would be good if we could do this using the new annotator.
    > This relates to [this issue](https://github.com/Living-with-machines/railspace/issues/1).
    > 
    > ### Describe your changes
    > 
    > This PR:
    > - Adds `filter_for` argument to the annotator class. You can now pass a dict (e.g. ``{"predicted_label":"railspace"}``) to the annotator and it will filter your patch_df for this col and value. 
    > 
    > For annotating predictions, I guess we just have labels of "correct" "incorrect" and "unsure" and then we can annotate as normal but using this filter for each label. 
    > 
    > Theres probably more to do - will think.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/354)
    > ### Summary
    > 
    > Update paper as per requests - https://github.com/openjournals/joss-reviews/issues/6168
    > 
    > Fixes #353 
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/356)
    > ### Summary
    > 
    > Bug fix for annotations since `filter_for` argument added.
    > 
    > ### Describe your changes
    > 
    > I made it so filtering doesn't happen if no `filter_for` is given. 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/350)
    > ### Summary
    > 
    > At the moment, MapReader has the option to use the context (i.e. central patch + surrounding patches) to improve the accuracy of the model. This is done with a custom model which has two branches and two separate input pipelines (one for patch only, one for context image). However, the context model code doesn't work very well. 
    > 
    > An idea to fix/improve this is to use a single branch model but use only the context image as input. Annotations will be at the patch level so the model should hopefully learn that annotations correspond to patterns in the centre of the context image (i.e. central patch) rather than the whole image. 
    >  
    > Addresses #17 
    > Addresses #287 
    > 
    > ### Describe your changes
    > 
    > - `square_cuts` argument is removed as option from `images.py` (Load subpackage). Patches at edges are now padded to create square patches so patch size is consistent across entire parent image.
    > - `annotator.py` now builds context image based on size of current patch so should work for patchify by meters
    > - `context_dataset` argument added when creating datasets in `load_annotations.py`. This calls a `create_context_datasets` method which creates a context dataset vs patch dataset.
    > - Context images in `PatchContextDataset` are now created by loading surrounding patches (as opposed to by no. of pixels either side). This is consistent with previous steps in the pipeline. 
    > - `PatchContextDataset` now returns ONLY context image (before it was context image + patch image). 
    > - `classifier.py` now works for context datasets and patch datasets (`context_classifier.py` is deleted as now redundant)
    > - Updated docs
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?

### Files

#### Added

- mapreader/process/post_process.py
- tests/sample_files/post_processing_patch_df.csv
- tests/test_classify/test_datasets.py
- tests/test_post_processing.py
- worked_examples/annotation/how-to-annotate-model-predictions.ipynb
- worked_examples/geospatial/context_classification_one_inch_maps/Pipeline.ipynb

#### Modified

- .github/workflows/mr_ci.yml
- docs/source/User-guide/Annotate.rst
- docs/source/User-guide/Classify/Train.rst
- docs/source/User-guide/Post-process.rst
- mapreader/__init__.py
- mapreader/annotate/annotator.py
- mapreader/classify/classifier.py
- mapreader/classify/custom_models.py
- mapreader/classify/datasets.py
- mapreader/classify/load_annotations.py
- mapreader/load/images.py
- paper/paper.md
- tests/sample_files/land_#rw#.csv
- tests/sample_files/model_test.pkl
- tests/sample_files/test.pkl
- tests/sample_files/test_annots.csv
- tests/sample_files/test_annots.tsv
- tests/sample_files/test_annots_append.csv
- tests/test_annotator.py
- tests/test_classify/test_annotations_loader.py
- tests/test_classify/test_classifier.py
- tests/test_geo_pipeline.py
- tests/test_import.py
- tests/test_load/test_images.py
- worked_examples/annotation/how-to-annotate-patches.ipynb

#### Removed

- mapreader/classify/classifier_context.py

## [v1.1.2](https://github.com/Living-with-machines/MapReader/releases/tag/v1.1.2) (2024-02-05)

### Summary

The changelog includes several key changes made through both commit messages and pull requests. 

#### Commit Messages:
- Various updates were made to delimiters and documentation.
- Several bugs were fixed, including issues related to the queue, sorting by criteria, and tests for random order of annotations.

#### Pull Requests:
1. **[PR #338](https://github.com/Living-with-machines/MapReader/pull/338)**: Implements changes from previous pull requests (#318, #326, #335, #337). Includes a checklist for self-review, passing tests, and documentation updates.
   
2. **[PR #345](https://github.com/Living-with-machines/MapReader/pull/345)**: Updates document links to be relative, ensuring that older versions of documents link correctly. Additionally, changes terminology from "rail_space" to "railspace" and fixes issue #320.

3. **[PR #347](https://github.com/Living-with-machines/MapReader/pull/347)**: Addresses a bug in the annotator that showed patches in a fixed order, altering it to a random order for better sampling. Also updates the loading of context to accommodate different pixel sizes in the dataset, fixing issue #346.

#### File Modifications:
Numerous files across documentation, annotation scripts, and test cases were modified or updated, while no files were added or removed. 

Overall, the updates focus on improving functionality, enhancing documentation, and resolving existing bugs.

### Commit messages

- change delimiter
- update delimiter
- update docs
- fix queue
- fix sortby
- fix tests for random order of annotations
- fix tests

### Pull requests

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/338)
    > ### Summary
    > 
    > This PR implements changes from PRs #318, #326 and #335 and #337.
    > See specifics in these PRs and related issues.
    > 
    > ### Describe your changes
    > 
    > Describe changes/updates you have made to the code.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/345)
    > ### Summary
    > 
    > Update links in docs to be relative. This means that if you are looking at old docs then clicking a link won't take you to v. latest. 
    > 
    > Update rail_space and rail space to railspace. 
    > 
    > Fixes #320
    > 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/347)
    > ### Summary
    > 
    > Currently there is a bug in the annotator where the patches are only shown in order. This would be better to be random to get better sampling for annotations. 
    > 
    > Fixes #346 
    > 
    > ### Describe your changes
    > 
    > - Changes the way in which the queue is generated 
    > - Updates loading of context incase patches have different pixel sizes throughout dataset (e.g. for "meters" patchify).
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?

### Files

#### Modified

- docs/source/Contribution-guide/Contribution-guide.rst
- docs/source/Contribution-guide/Documentation.rst
- docs/source/Contribution-guide/Worked-examples.rst
- docs/source/Input-guidance.rst
- docs/source/User-guide/Annotate.rst
- docs/source/User-guide/Classify/Classify.rst
- docs/source/User-guide/Classify/Infer.rst
- docs/source/User-guide/Classify/Train.rst
- docs/source/User-guide/Download.rst
- docs/source/User-guide/Load.rst
- docs/source/User-guide/User-guide.rst
- docs/source/Worked-examples/one_inch_pipeline.ipynb
- mapreader/annotate/annotator.py
- tests/sample_files/annotation_tasks.yaml
- tests/test_annotator.py
- worked_examples/annotation_tasks.yaml
- worked_examples/geospatial/classification_one_inch_maps/Pipeline.ipynb
- worked_examples/geospatial/classification_one_inch_maps/annotation_tasks.yaml
- worked_examples/geospatial/workshop_june_2023/Workshop_June2023.ipynb
- worked_examples/geospatial/workshop_june_2023/Workshop_June2023_empty.ipynb
- worked_examples/non-geospatial/classification_mnist/annotation_tasks_mnist.yaml

## [v1.1.1](https://github.com/Living-with-machines/MapReader/releases/tag/v1.1.1) (2024-01-08)

### Summary

The changelog captures a series of updates and fixes made to the MapReader project. Key actions include:

1. **Code Fixes and Improvements**:
   - Resolved issues with saving to GeoJSON and corrected coordinate processing.
   - Added methods for saving parent images as GeoTIFFs and handling single-band images.
   - Improved shape calculation for images.

2. **Testing Enhancements**:
   - Introduced tests for coordinate saving and grayscale images, ensuring accurate outputs in relation to QGIS comparisons.
   - Made various updates and additions to existing tests.

3. **Documentation Updates**:
   - Added a draft of a JOSS paper and incorporated corrections based on reviewer feedback.
   - Updated user guides and documentation files for clarity and accuracy.

4. **Version Control and CI/CD**:
   - Merged several branches to integrate improvements and fixes.
   - Updated GitHub Actions for continuous integration, including adding a Codecov badge.

5. **Minor Adjustments**:
   - Included small changes, cleaned up unnecessary codes, and refined metadata files.

Overall, these updates enhance the functionality, testing coverage, and documentation of the project, ensuring a more robust and user-friendly experience.

### Commit messages

- fix save to geojson
- add literal eval for list/tuple columns
- add method to save parents as geotiffs
- fix patch coords
- add saving coords from grid_bb
- add paper
- Small changes in the first paragraph; reordered tags and added DL
- Add an example about 16K map sheets
- minor
- Update paper.md per Kasra's comments
- Merge remote-tracking branch 'origin/fix_save_to_geojson' into 279-test-coord-saving
- add tests for coord saving (downloader)
- add approx for coords
- minor updates + v number
- Update mr_ci.yml
- update mr_ci.yml
- update mr_ci.yml
- Update mr_ci.yml
- add codecov badge
- Rename Contributors.md to contributors.md
- update metadata files
- add and update tests
- add more tests
- remove unnecessary literal_evals
- update docs
- calc shape from height, width and channels explicitly and allow for single band images
- fix links
- remove kwargs
- remove error if no url
- allow for image_id to be column 0
- add ClassifierContainer imports to docs
- add saving of one band geotiffs
- Merge branch 'dev' into 279-test-coord-saving
- Merge branch 'dev' into 331-hwc-bug
- add tests for grayscale images
- Merge branch 'dev' into dev_annotator
- remove fail on no url col
- add missing tests

### Pull requests

- [kmcdono2](https://github.com/Living-with-machines/MapReader/pull/321)
    > ### Summary
    > 
    > Fixes @kasra-hosseini feedback in #316 
    > 
    > ### Describe your changes
    > 
    > - add citation to JVC paper as reference for statement about map digitization
    > - add comment about documentation in conclusion
    > 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [ ] check paper still compiles
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/316)
    > ### Summary
    > 
    > This PR adds the JOSS paper and references file to the repo in a directory called `paper`.
    > It also adds a GH action which builds a draft of the paper - we should turn this off once we are done making changes!
    > 
    > Fixes #257 
    > 
    > ## To do still
    > 
    > - [x] Check output of GH action
    > - [x] Fill in title, AHRC grant no., anything else that is currently "XXXX".
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/328)
    > To add codecov to the repo.
    > Addresses #314 
    > 
    > - [ ] Add badge
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/330)
    > This PR adds the codecov badge to the repo.
    > Fixes #314 
    > 
    > Note: Other badge formats are available - see https://app.codecov.io/gh/Living-with-machines/MapReader/settings/badge for info.
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/336)
    > Fixes links in the Classify Train/Infer pages.
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/318)
    > ### Summary
    > 
    > At the moment, the code can output patches as geojson and geotiffs. These methods rely on coordinates being saved in the patch/parent metadata.
    > 
    > Problems with this come from:
    > - Coordinates are saved as a list in the patch/parent dataframes (`[minx, miny, maxx, maxy]`) so when reading from a csv, these coords are read as a string and not a python list object. This is a problem when indexing the coord list later on. 
    > - In an earlier version of mapreader, the parent coordinates we calculated incorrectly. If these coords are in your parent dataframe it is not possible to recalculate using other info saved for hte parent image (i.e. the grid bounding box used for downloading)
    > - The save to geojson method uses a shapely polygon (generated from coordinates) for saving geometry instead of the coordinates. When reading from a csv, the polygon is read as a string not a polygon object.
    > 
    > ### Describe your changes
    > 
    > This PR:
    > - Adds method to apply `literal_eval` to all columns in dataframe using try/except. This fixes problems with tuples/coordinates/etc being read as strings. i.e. fixes problem with coords list.
    > - Adds method of regenerating coordinates from grid bounding boxes. i.e. to fix previously incorrect coords from bug in earlier version of MapReader.
    > - Recreates shapely polygon from string if needed
    > - Adds method for saving parent images as geotiffs. This is helpful because both parent/patch images can then be loaded into QGIS or other geospatial software and compared to a basemap to check that coords are as expected.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    > - [x] Update relevant docs
    > 
    > To do:
    > - [x] Add method to ensure single band images are saved correctly for `save_parents_as_geotiffs()`
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/335)
    > Fixes #331 
    > 
    > ### Summary
    > 
    > This PR changes the calculation of image shape such that the height, width and no. of channels are taken explicitly from the image not the array shape. This means that single band images get channels =1 rather than no value for channels.
    > 
    > ### Describe your changes
    > 
    > - Change how shape is calculated
    > - Update plotting to allow for single band images
    > - Updates geotiff saving to allow for single band images
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/326)
    > ### Summary
    > 
    > This PR adds tests for coordinate saving. 
    > These coords have been checked by saving maps as geotiffs and comparing in QGIS against an OSM basemap. 
    > 
    > Fixes #279 
    > 
    > This PR should be merged after #318 as it has taken some changes from there. 
    > `git merge main` before merging this PR.
    > 
    > ### Describe your changes
    > 
    > - Coords are tested for both zoom level 10 and 14.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/337)
    > ### Summary
    > 
    > Fixes #332 
    > Fixes #333 
    > 
    > ### Describe your changes
    > 
    > This PR:
    > - removes kwargs from the Annotator's `__init__()` method. This means that if you spell something wrong or pass an unrecognised arg you will get an error.
    > - removes error if URL not found in parent df.
    > - resets index when loading annoations, this means if your annotations df has "image_id" in col 0 then it will not error.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    > - [x] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?

### Files

#### Added

- .github/workflows/joss-draft-pdf.yml
- paper/paper.bib
- paper/paper.md
- tests/sample_files/cropped_L.png

#### Modified

- .github/workflows/mr_ci.yml
- README.md
- docs/source/User-guide/Annotate.rst
- docs/source/User-guide/Classify/Classify.rst
- docs/source/User-guide/Classify/Train.rst
- docs/source/User-guide/Load.rst
- mapreader/annotate/annotator.py
- mapreader/classify/load_annotations.py
- mapreader/load/images.py
- tests/sample_files/ts_downloaded_maps.csv
- tests/sample_files/ts_downloaded_maps.tsv
- tests/sample_files/ts_downloaded_maps.xlsx
- tests/test_annotator.py
- tests/test_load/test_geo_utils.py
- tests/test_load/test_images.py
- tests/test_sheet_downloader.py

## [v1.1.0](https://github.com/Living-with-machines/MapReader/releases/tag/v1.1.0) (2023-12-14)

### Summary

The changelog includes several commit messages and pull requests detailing updates and fixes to the project. Key updates include:

- **Fixes**: Addressed issues with reprojecting geographic information, clarified title from "Project CV" to "Project Curriculum Vitae," and resolved a bug related to the variable `size_in_m` being used before assignment.
- **Documentation**: Added contributor documentation and updated user guide sections, particularly with contributions from Tim Hobson.
- **New Features**: Introduction of a new `Annotator` class aimed at enhancing the user experience in annotating patches, alongside checks to ensure joining and filtering of annotations are seamless.
- **File Modifications**: Various files, particularly documentation guides, have been modified. The file `how_to_contribute_to_docs.md` was removed.

Overall, the changes focus on improving functionality, fixing bugs, and enhancing documentation clarity.

### Commit messages

- fix reproject geo info
- add contributor docs
- change false to none
- Update docs/source/User-guide/Annotate.rst    Co-authored-by: Tim Hobson <thobson88@gmail.com>
- Update docs/source/User-guide/Annotate.rst    Co-authored-by: Tim Hobson <thobson88@gmail.com>
- rename as "Project Curriculum Vitae"    Fixes #316 comment about Project CV being unclear (re: computer vision vs. curriculum vitae)

### Pull requests

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/317)
    > ### Summary
    > 
    > Fixes #313 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Ensure submission passes current tests
    > - [x] Update relevant docs
    > - [x] Check on readthedocs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/277)
    > `size_in_m`  variable was outside if statement and sometimes references before being set.
    > 
    > Tests should be added as part of #279 
- [kallewesterling](https://github.com/Living-with-machines/MapReader/pull/173)
    > Attempting to create a new `Annotator` class for MapReader, which will provide a less buggy experience for the user who is trying to annotate their patches.
    > 
    > Feature requests:
    > - [X] ensure joining on existing annotations works
    > - [X] ensure not re-annotating
    > - [X] showing context
    > - [x] Add in `sortby` keyword, for sorting by a different column (existed in prior version)
    > - [x] Add a `min_mean_pixel` and `max_mean_pixel` for filtering data shown to you (existed in prior version)
    > - [x] display URL to NLS map
    > - [x] keep patch filepaths in annotations csv output
    > - [x] keep label names (either instead of or in addition to label indices) in annotations csv output.
    > - [X] margin
    > - [x] "next random patch"
    > - [ ] keyboard shortcuts
    > - [ ] batch sizing
    > - [ ] restrict to bounding box
    > - [ ] consider `load` and `dump` method/s (for, for instance LabelStudio input/output etc.)
    > - [x] Make it obvious that 'next' and 'previous' are next random patch not like moving left/right on context image
- [kmcdono2](https://github.com/Living-with-machines/MapReader/pull/322)
    > 
    > 
    > ### Summary
    > 
    > Fixes #316 comment about Project CV being unclear (re: computer vision vs. curriculum vitae)
    > 
    > ### Describe your changes
    > 
    > Changed title in header from "Project CV" to "Project Curriculum Vitae"
    > 
    > NB: I did not change the name of the actual file.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [ ] Self-review code
    > - [ ] Ensure submission passes current tests
    > - [ ] Add tests
    > - [ ] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?

### Files

#### Modified

- docs/source/Contribution-guide/Contribution-guide.rst
- docs/source/Project-cv.rst
- docs/source/User-guide/Annotate.rst
- mapreader/load/geo_utils.py

#### Removed

- how_to_contribute_to_docs.md

## [v1.0.7](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.7) (2023-12-14)

### Summary

The changelog outlines a series of updates primarily focused on the development and enhancement of a new `Annotator` class in the `MapReader` project. Key changes include:

1. **New Features and Enhancements**:
   - Introduction of the `Annotator` class with typing, metadata management, and several new functionalities such as options for progress bars and filtering based on pixel values.
   - Added methods and parameters for fine-tuning annotation processes, including a new `show_context` option and an auto-resizing feature for patch images.
   - Numerous documentation updates and examples to aid user understanding.

2. **Bug Fixes**:
   - A variety of minor bugs and typos were addressed, improving overall stability and user experience. 

3. **Refactoring and Code Organization**:
   - Code restructuring was undertaken to enhance maintainability, including moving the `Annotator` class, altering UI elements, and implementing new filtering capabilities.

4. **Documentation and Contributor Engagement**:
   - Updates to README and various documentation files to improve clarity and user guidance, along with the introduction of an all-contributors bot to acknowledge contributions.

5. **Test Additions**:
   - New tests were added to ensure the reliability of the updates and to support ongoing development.

6. **Files Changed**:
   - Several files were modified, particularly in the documentation and the core functionality of the `annotator.py` script, alongside the introduction of new example notebooks and test scripts.

Overall, this changelog reflects a comprehensive effort to enhance the functionality, usability, and documentation of the `MapReader` project.

### Commit messages

- First commit of new `Annotator` class
- Adding an example
- Fixing a tiny bug
- Adding in metadata + fixing docstring + little bug
- Formatting
- Dropping unnecessary and conflicting import
- Adding in some typing
- Adding a missing parameter (`stop_at_last_example`)
- Clarifying `annotations_file` attribute
- Spelling mistake in typing
- Adding `show_context` option
- Adding a progress bar
- moving `Annotator` from `annotate.utils` to `annotate`
- Adding in a few notes in the docs
- More docs additions
- Keep patch filepaths + keep label names in output
- Adding `sortby` keyword argument
- Adding docstring for `sortby`
- Adding ability to filter (like `min_mean_pixel` and `max_mean_pixel`)
- Adding showing of `url` (and fixing some bugs)
- Changing look of progress bar
- Refactoring the code
- Refactoring, more UI, better filtering
- Bugfix
- Changing UI a bit further
- Adding examples to `annotate` method
- Filling out example better
- Changing `min_values` + `max_values` to `mean_pixel_RGB` in example for `annotator` method
- Refactoring as queue + UI design
- Adding `metadata_delimiter` keyword argument
- Adding debug messaging
- Fix bug
- Renaming frames + ensuring annotation_dir exists
- Adding in a TODO
- Spelling fix
- Fix typo again
- Fixing annoying spelling error
- Adding TODO
- Renaming `"changed"` column `"annotated"`
- Adding auto-resizing of patch images to 100px
- Fixing detail
- Adding margin as keyword arg to `annotate`
- Name set to next/next random patch depending on settings
- Updated `Annotator`
- Adding in a TODO
- Dropping unnecessary f-string
- Create annotations_dir
- Adding worked example for annotations
- Cleaning notebook
- Trying to resolve display issue
- update section headers
- Update README.md (add contributors)
- Update README.md
- Create .all_contributors.rc
- docs: update README.md [skip ci]
- docs: create .all-contributorsrc [skip ci]
- docs: update README.md [skip ci]
- docs: update .all-contributorsrc [skip ci]
- docs: update README.md [skip ci]
- docs: update .all-contributorsrc [skip ci]
- docs: update README.md [skip ci]
- docs: update .all-contributorsrc [skip ci]
- docs: update README.md [skip ci]
- docs: update .all-contributorsrc [skip ci]
- Update README.md
- Update README.md
- Update .all-contributorsrc
- Update README.md
- Update README.md
- docs: update README.md [skip ci]
- docs: update .all-contributorsrc [skip ci]
- update notebook
- rename annotator file
- docs: update README.md [skip ci]
- docs: update .all-contributorsrc [skip ci]
- docs: update README.md [skip ci]
- docs: update .all-contributorsrc [skip ci]
- docs: update README.md [skip ci]
- docs: update .all-contributorsrc [skip ci]
- docs: update README.md [skip ci]
- docs: update .all-contributorsrc [skip ci]
- docs: update README.md [skip ci]
- docs: update .all-contributorsrc [skip ci]
- docs: update README.md [skip ci]
- docs: update .all-contributorsrc [skip ci]
- docs: update README.md [skip ci]
- docs: update .all-contributorsrc [skip ci]
- docs: update README.md [skip ci]
- docs: update .all-contributorsrc [skip ci]
- Update .all-contributorsrc
- add kwargs as normal args (not tested)
- add literal_eval for reading list/tuple columns
- fix sortby and min/max values
- update setup.py to work with jupyter notebook/lab
- update notebook
- update docs
- add geopandas to dependencies
- update test imports
- add tests and minor update to annotator.py
- fix sorting
- add tests for delimiter
- update docstrings
- add filter_for arg to annotator
- minor update to annotator
- add surrounding arg to docs
- add ``resize_to`` kwarg to resize small patches
- updates docs for resize_to
- Merge branch 'kallewesterling/issue166' into analyse_preds
- change how max_size is set in lieu of resize_to param
- Merge branch 'kallewesterling/issue166' into analyse_preds

### Pull requests

- [kallewesterling](https://github.com/Living-with-machines/MapReader/pull/181)
    > Pulling latest edits from `main`
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/290)
    > ### Summary
    > 
    > Fixes #289 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/297)
    > ### Summary
    > 
    > Add contributors bot
- [allcontributors[bot]](https://github.com/Living-with-machines/MapReader/pull/298)
    > Adds @rwood-97 as a contributor for code.
    > 
    > This was requested by rwood-97 [in this comment](https://github.com/Living-with-machines/MapReader/pull/296#issuecomment-1829426830)
    > 
    > [skip ci]
- [allcontributors[bot]](https://github.com/Living-with-machines/MapReader/pull/299)
    > Adds @kmcdono2 as a contributor for research, ideas.
    > 
    > This was requested by rwood-97 [in this comment](https://github.com/Living-with-machines/MapReader/issues/255#issuecomment-1829433856)
    > 
    > [skip ci]
- [allcontributors[bot]](https://github.com/Living-with-machines/MapReader/pull/300)
    > Adds @dcsw2 as a contributor for research.
    > 
    > This was requested by rwood-97 [in this comment](https://github.com/Living-with-machines/MapReader/issues/255#issuecomment-1829433856)
    > 
    > [skip ci]
- [allcontributors[bot]](https://github.com/Living-with-machines/MapReader/pull/301)
    > Adds @kasra-hosseini as a contributor for code.
    > 
    > This was requested by rwood-97 [in this comment](https://github.com/Living-with-machines/MapReader/issues/255#issuecomment-1829439205)
    > 
    > [skip ci]
- [allcontributors[bot]](https://github.com/Living-with-machines/MapReader/pull/302)
    > Adds @rwood-97 as a contributor for doc.
    > 
    > This was requested by rwood-97 [in this comment](https://github.com/Living-with-machines/MapReader/issues/255#issuecomment-1829440342)
    > 
    > [skip ci]
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/303)
    > ### Summary
    > 
    > Describe the problem you're trying to fix in this pull request.
    > Please reference any related issue and use fixes/close to automatically close them, if pertinent. For example: "Fixes #58", or "Addresses (but does not close) #238". -->
    > 
    > Fixes #<NUM>
    > 
    > ### Describe your changes
    > 
    > Describe changes/updates you have made to the code.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [ ] Self-review code
    > - [ ] Ensure submission passes current tests
    > - [ ] Add tests
    > - [ ] Update relevant docs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [allcontributors[bot]](https://github.com/Living-with-machines/MapReader/pull/304)
    > Adds @kallewesterling as a contributor for code, doc.
    > 
    > This was requested by rwood-97 [in this comment](https://github.com/Living-with-machines/MapReader/issues/255#issuecomment-1829795774)
    > 
    > [skip ci]
- [allcontributors[bot]](https://github.com/Living-with-machines/MapReader/pull/305)
    > Adds @ChrisFleet as a contributor for data.
    > 
    > This was requested by rwood-97 [in this comment](https://github.com/Living-with-machines/MapReader/issues/255#issuecomment-1835768982)
    > 
    > [skip ci]
- [allcontributors[bot]](https://github.com/Living-with-machines/MapReader/pull/306)
    > Adds @kasparvonbeelen as a contributor for ideas, review, research.
    > 
    > This was requested by rwood-97 [in this comment](https://github.com/Living-with-machines/MapReader/issues/255#issuecomment-1835770807)
    > 
    > [skip ci]
- [allcontributors[bot]](https://github.com/Living-with-machines/MapReader/pull/307)
    > Adds @kmcdono2 as a contributor for doc, eventOrganizing, projectManagement, review, talk, tutorial.
    > 
    > This was requested by rwood-97 [in this comment](https://github.com/Living-with-machines/MapReader/issues/255#issuecomment-1835773542)
    > 
    > [skip ci]
- [allcontributors[bot]](https://github.com/Living-with-machines/MapReader/pull/308)
    > Adds @dcsw2 as a contributor for ideas, talk, doc, eventOrganizing.
    > 
    > This was requested by rwood-97 [in this comment](https://github.com/Living-with-machines/MapReader/issues/255#issuecomment-1835775445)
    > 
    > [skip ci]
- [allcontributors[bot]](https://github.com/Living-with-machines/MapReader/pull/309)
    > Adds @kasra-hosseini as a contributor for ideas, research, review, talk.
    > 
    > This was requested by rwood-97 [in this comment](https://github.com/Living-with-machines/MapReader/issues/255#issuecomment-1835779186)
    > 
    > [skip ci]
- [allcontributors[bot]](https://github.com/Living-with-machines/MapReader/pull/310)
    > Adds @rwood-97 as a contributor for ideas, talk, tutorial, review, maintenance, research.
    > 
    > This was requested by rwood-97 [in this comment](https://github.com/Living-with-machines/MapReader/issues/255#issuecomment-1835780534)
    > 
    > [skip ci]
- [allcontributors[bot]](https://github.com/Living-with-machines/MapReader/pull/311)
    > Adds @kallewesterling as a contributor for maintenance, review, talk.
    > 
    > This was requested by rwood-97 [in this comment](https://github.com/Living-with-machines/MapReader/issues/255#issuecomment-1835782492)
    > 
    > [skip ci]
- [allcontributors[bot]](https://github.com/Living-with-machines/MapReader/pull/312)
    > Adds @andrewphilipsmith as a contributor for code, doc, mentoring, review.
    > 
    > This was requested by rwood-97 [in this comment](https://github.com/Living-with-machines/MapReader/issues/255#issuecomment-1835784305)
    > 
    > [skip ci]

### Files

#### Added

- .all-contributorsrc
- .all_contributors.rc
- mapreader/annotate/annotator.py
- tests/test_annotator.py
- worked_examples/annotation/how-to-annotate-patches.ipynb

#### Modified

- README.md
- docs/source/Beginners-info.rst
- docs/source/Contribution-guide/Contribution-guide.rst
- docs/source/Contribution-guide/GitHub-guide.rst
- docs/source/Developers-guide.rst
- docs/source/Input-guidance.rst
- docs/source/Install.rst
- docs/source/User-guide/Annotate.rst
- docs/source/User-guide/Classify/Classify.rst
- docs/source/Worked-examples/Worked-examples.rst
- mapreader/__init__.py
- mapreader/annotate/utils.py
- setup.py
- tests/test_import.py

## [v1.0.6](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.6) (2023-11-22)

### Summary

The changelog outlines updates made to the MapReader project, including various commit messages and pull requests. Key updates include the addition of an 'images_dir' argument for loading annotations, modifications to error handling for broken paths, enhancements to testing frameworks, changes in deployment triggers for conda and PyPI to respond to tagged commits, documentation improvements, and the introduction of new files like citation and contributor documents. 

Significant pull requests address issues such as correcting metadata extraction, refining image downloading logic, and enhancing data processing capabilities. Other improvements include the implementation of pre-commit configurations for consistent formatting, updates in the documentation structure, and code adjustments for accuracy in data handling. Overall, the updates reflect a continuous effort to enhance functionality, performance, and maintainability of the MapReader software.

### Commit messages

- add 'images_dir' argument to load_annotations
- update error messages
- add function to check patch paths
- raise error if no annotations remain
- update tests
- update tests (fix error)
- Changes conda GH Action to only deploy on tagged commits or when manually triggered
- add test_classifier.py update
- print full (abs) path for broken_files.txt
- fix pygeos vs shapely warning
- us os.path.join to update paths
- fix print full (abs) path for broken_files.txt
- fix problem of using df_test =0
- error if remove_broken=False and broken paths exist
- add citation.cff file
- add K Westerling as author
- Update CITATION.cff    fix formatting + add Kalle
- update tests
- fix indentation error
- fix test
- Merge branch '219-annotation-file-paths' of https://github.com/Living-with-machines/MapReader into 219-annotation-file-paths
- Update Install.rst
- add project cv and events page
- Update Project-cv.rst
- Update Project-cv.rst
- Update Project-cv.rst
- Update Project-cv.rst
- Update Project-cv.rst
- katie updates
- Update Project-cv.rst    Removed some repetition from the first section
- Update Project-cv.rst
- fix typos
- add DOI badge
- add citation info
- split Code of conduct
- Update ways_of_working.md
- Create Contributors.md
- add geo pipeline tests
- Update images.py
- add pre-commit and ruff configs
- pre-commit run all
- Update Contributors.md
- Update ways_of_working.md
- Update README.md
- Update Contributors.md
- check both upper and lower corners when finding tilesize
- raise error is both corners are missing
- ensure download doesn't fail if maps are not found
- only drop absolute duplicates
- Update Contributors.md    add DVS, OV, RA, and JL
- fix coordinate saving
- Merge branch 'dev_download' of https://github.com/Living-with-machines/MapReader into dev_download
- exclude worked examples from pre-commit
- add __init__.py and test_import.py to excludes
- add create_dataloader method to PatchDataset
- remove backslashes
- add pyupgrade
- run pyupgrade
- run all
- add option to load ClassifierContainer with no dataloaders
- add default for dataloaders arg
- only require criterion for training/validation
- fix tests
- fix notebooks
- update docs
- fix adding of dataloaders if load_path also passed
- update docs - split into train/infer
- fix file paths
- fix tests
- fix typo (fix tests)
- i actually tested it this time
- Add docs for inference only
- add tests
- fix typo
- update about docs
- Update About.rst
- fix drop duplicates
- run pre-commit
- update date saving for extract_published_dates
- Allow user to select metadata to save
- add try/excpt to sheet downloader
- update/add tests
- update geo pipeline test
- Update PULL_REQUEST_TEMPLATE.md
- add docs

### Pull requests

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/228)
    > ### Summary
    > 
    > Currently, we are saving image paths in our annotations output (annotations_df contains columns for: image_id, image_path and label). This makes loading images easier in the `classify` subpackage as we do not need to re-locate our images by specifying the file paths. 
    > However, in instances where the patches directory is moved between `annotate` and `classify` (e.g. you annotate on your laptop and move to VM for training or if multiple people are annotating), this is an issue.
    > 
    > This PR addresses this by adding a `images_dir` argument to the `AnnotationsLoader().load()` method which allows you to specify and update the image paths when loading in your annotations.
    > It also adds a function to check the validity of image paths when loading annotations (as per #7) and remove annotations with broken file paths.
    > 
    > Fixes #219 
    > Fixes #7
    > 
    > ### Describe your changes
    > 
    > - Add 'images_dir' argument.
    > - Update error messages.
    > - Add '_check_file_paths()' method
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [andrewphilipsmith](https://github.com/Living-with-machines/MapReader/pull/246)
    > ### Summary
    > 
    > This PR fixes (or contributes to fixing) https://github.com/Living-with-machines/MapReader/issues/162
    > 
    > It seems that this error was caused by the relevant conda-forage account using more than its allocated storage space.
    > 
    > ### Describe your changes
    > 
    > This PR alters the deployment action to only publish to conda-forge on tagged commits (or when manually triggered). This is the same behaviour as for PyPI.
    > 
    > Additionally, old versions of MapReader have been manually deleted from conda-forge.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [ ] Ensure submission passes current tests
    > - [x] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/253)
    > ### Summary
    > 
    > Adding project cv and events to docs
    > 
    > Fixes #231 
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/256)
    > ### Summary
    > 
    > This PR splits the Code of conduct from the Contribution guide. 
    > See https://mapreader.readthedocs.io/en/rw_docs/index.html#
    > 
    > It also splits the Contributors from the Ways of working file.
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/258)
    > ### Summary
    > 
    > This PR adds a 'test_geo_pipeline.py' file which goes through the full pipeline and runs the 'train' and 'inference' parts of the classify package.
    > Further dev would probably be useful here but is a good first step.
    > 
    > Addresses #171 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/269)
    > ### Summary
    > 
    > At the moment to get the tile size, the tile merger looks at the size of the tile found in the bottom left corner of the image. This is problematic if this tile is missing.
    > 
    > This PR changes it so that it tries the lower left corner, then if its not found, tries the upper right corner and then if its not found, raises an error.
    > 
    > Addresses #239. 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/270)
    > ### Summary
    > 
    > In PR #269 I changed the code so that the downloader errored if both the bottom left and upper right corners of the map were missing.
    > This PR changes that so it so that if this error occurs, instead of erroring, MapReader prints a message saying that the download of xxx map was unsuccessful and then continues on to the next map.
    > 
    > Again, relates to #239 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/276)
    > ### Summary
    > 
    > This PR ensures that coordinates of the downloaded maps are saved correctly. 
    > 
    > ![image](https://github.com/Living-with-machines/MapReader/assets/72076688/0ce4140b-e58d-485e-9dd1-0f38b13c2ae0)
    > When downloading tiles, the coordinates of the tile at (1,1) correspond to the bottom left corner of the tile. This is the same as the upper right corner of tile (0,0).
    > 
    > If tile (3,3) makes up the upper right corner of the map image we have downloaded, then the coordinate of the upper right corner of the map image is not the coordinate (3,3) but actually (4,4). 
    > This needed updating in the code as the coordinates that were previously saved were off by one tile index.
    > 
    > ### Describe your changes
    >  Coordinates of NE corner of the map is now tile index +1 .
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/278)
    > ### Summary
    > 
    > This PR adds pre-commit to the repo to ensure formatting is consistent.
    > Fixes #264 
    > 
    > ### Describe your changes
    >   
    > Add pre-commit and ruff config.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/283)
    > ### Summary
    > 
    > Describe the problem you're trying to fix in this pull request.
    > Please reference any related issue and use fixes/close to automatically close them, if pertinent. For example: "Fixes #58", or "Addresses (but does not close) #238". -->
    > 
    > Fixes #<NUM>
    > 
    > ### Describe your changes
    > 
    > - Add the input/output pictures to the about page
    > - Rename 'classify' part of the pipeline diagram to 'train'
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Check on rw_docs version of MapReader readthedocs
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/285)
    > ### Summary
    > 
    > At the moment there is an error due to one of our df columns containing a list.
    > The fix for this is to convert everything to strings, find the indices of duplicates to drop, then drop those from the original dataframe.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/280)
    > ### Summary
    > 
    > This PR fixes #275 in two ways:
    > - Adds method to create a dataloader from a `PatchDataset`.
    > - Adds the ability to create a `ClassifierContainer` without defining a dataloader. Datasets can then be added using the `load_dataset()` method.
    > 
    > Fixes #275
    > 
    > ### Describe your changes
    >   
    > - Add `create_dataloaders()` method to `PatchDataset`
    > - Require `model` and `labels_map` or `load_path`
    > - Only require criterion for training/validation
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    > - [x] Update docs
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/291)
    > ### Summary
    > 
    > At the moment, regardless of how much info you have in your metadata.json file (e.g. [here](https://github.com/Living-with-machines/MapReader/blob/main/worked_examples/geospatial/NLS_metadata/metadata_OS_One_Inch_GB_WFS_light.json) and [here](https://github.com/Living-with-machines/MapReader/blob/main/worked_examples/geospatial/NLS_metadata/metadata_OS_Six_Inch_GB_WFS_light.json), MapReader only transfers "name", "url", "coordinates", "crs", "published_date" and "grid_bb" to the metadata.csv.
    > 
    > As per Issue #272 - It would also be nice to transfer additional data (e.g. county, survey date, IIIF url) to the metadata.csv. This PR fixes this by enabling users to pass a "metadata_to_save" dictionary to the download_maps methods as a kwarg so that they can select extra fields to save. E.g. {"county":["properties","COUNTY"]} will add saving the county info to the metadata.csv.
    > 
    > Behaviour is now:
    > ```
    > kwargs = {"metadata_to_save":{"county":["properties","COUNTY"], "another example":"metadata key"}}
    > my_ts.download_all_map_sheets(path_save, metadata_fname, **kwargs)
    > ```
    > 
    > This will result in a metadata.csv with columns of: 
    > - "name", "url", "coordinates", "crs", "published_date", "grid_bb",  (i.e. same as before, plus..)
    > - "county"
    > - "another example".
    > 
    > Another relate problem is Issue #273 - "published_date" is extracted from the "WFS_TITLE" of the map metadata. This normally looks something like "Anglesey I.NE, Surveyed: 1877, Published: 1888" for the NLS maps but might be different for different libraries (e.g. could just say "Anglesey 1.NE" or something completely different). [Some NLS metadata.json files](https://github.com/Living-with-machines/MapReader/blob/main/worked_examples/geospatial/NLS_metadata/metadata_OS_Six_Inch_GB_WFS_light.json) also contain fields like "PUB_END" or "YEAR" which would be nice to use instead. This PR enables the user to specify a ``date_col`` when extracting the published dates and to override the default regex search in "WFS_TITLE" so that if a better column is present then we just use that.
    > 
    > This can also be used as a kwarg in the download maps methods as above:
    > ```
    > kwargs = {"metadata_to_save":{"county":["properties","COUNTY"], "another example":"metadata key"}, 
    > "date_col":["properties", "YEAR"]} 
    > my_ts.download_all_map_sheets(path_save, metadata_fname, **kwargs)
    > ```
    > 
    > Then metadata.csv will have dates taken from ``dict["properties"]["YEAR"]`` instead of WFS title.
    > 
    > Fixes #272 
    > Fixes #273
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    > 
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    > 
    > ### Reviewer checklist
    > 
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [x] Everything looks ok?

### Files

#### Added

- .pre-commit-config.yaml
- .ruff.toml
- CITATION.cff
- Contributors.md
- docs/source/Coc.rst
- docs/source/Events.rst
- docs/source/Project-cv.rst
- docs/source/User-guide/Classify/Classify.rst
- docs/source/User-guide/Classify/Infer.rst
- tests/sample_files/land_#rw#.csv
- tests/test_geo_pipeline.py

#### Modified

- .github/PULL_REQUEST_TEMPLATE.md
- .github/workflows/mr_ci.yml
- .github/workflows/mr_pip_ci.yml
- .github/workflows/publish-to-conda-forge.yml
- .github/workflows/publish-to-pypi.yml
- .readthedocs.yaml
- README.md
- docs/requirements.txt
- docs/source/About.rst
- docs/source/Beginners-info.rst
- docs/source/Contribution-guide/Code.rst
- docs/source/Contribution-guide/Contribution-guide.rst
- docs/source/Contribution-guide/Documentation.rst
- docs/source/Contribution-guide/GitHub-guide.rst
- docs/source/Contribution-guide/Worked-examples.rst
- docs/source/Developers-guide.rst
- docs/source/Input-guidance.rst
- docs/source/Install.rst
- docs/source/User-guide/Annotate.rst
- docs/source/User-guide/Download.rst
- docs/source/User-guide/Load.rst
- docs/source/User-guide/Post-process.rst
- docs/source/User-guide/User-guide.rst
- docs/source/Worked-examples/Worked-examples.rst
- docs/source/Worked-examples/mnist_pipeline.ipynb
- docs/source/Worked-examples/one_inch_pipeline.ipynb
- docs/source/Worked-examples/plant_pipeline.ipynb
- docs/source/conf.py
- docs/source/figures/pipeline_explained.png
- docs/source/index.rst
- mapreader/_version.py
- mapreader/annotate/utils.py
- mapreader/classify/classifier.py
- mapreader/classify/classifier_context.py
- mapreader/classify/custom_models.py
- mapreader/classify/datasets.py
- mapreader/classify/load_annotations.py
- mapreader/download/data_structures.py
- mapreader/download/downloader.py
- mapreader/download/downloader_utils.py
- mapreader/download/sheet_downloader.py
- mapreader/download/tile_loading.py
- mapreader/download/tile_merging.py
- mapreader/load/geo_utils.py
- mapreader/load/images.py
- mapreader/load/loader.py
- mapreader/process/process.py
- mapreader/utils/compute_and_save_stats.py
- mapreader/utils/slice_parallel.py
- setup.py
- tests/sample_files/annotation_tasks.yaml
- tests/sample_files/test_annots.csv
- tests/sample_files/test_annots.tsv
- tests/sample_files/test_json.json
- tests/sample_files/test_json_epsg3857.json
- tests/test_classify/test_annotations_loader.py
- tests/test_classify/test_classifier.py
- tests/test_load/test_geo_utils.py
- tests/test_load/test_images.py
- tests/test_load/test_images_load_parents.py
- tests/test_load/test_loader.py
- tests/test_load/test_loader_load_patches.py
- tests/test_sheet_downloader.py
- versioneer.py
- ways_of_working.md
- worked_examples/annotation_tasks.yaml
- worked_examples/geospatial/NLS_metadata/README.md
- worked_examples/geospatial/NLS_metadata/metadata_OS_One_Inch_GB_WFS_light.json
- worked_examples/geospatial/NLS_metadata/metadata_OS_Six_Inch_GB_WFS_light.json
- worked_examples/geospatial/README.md
- worked_examples/geospatial/classification_one_inch_maps/Pipeline.ipynb
- worked_examples/geospatial/classification_one_inch_maps/annotation_tasks.yaml
- worked_examples/geospatial/workshop_june_2023/Workshop_June2023.ipynb
- worked_examples/geospatial/workshop_june_2023/Workshop_June2023_empty.ipynb
- worked_examples/non-geospatial/README.md
- worked_examples/non-geospatial/classification_mnist/Pipeline.ipynb
- worked_examples/non-geospatial/classification_mnist/annotation_tasks_mnist.yaml
- worked_examples/non-geospatial/classification_plant_phenotype/Pipeline.ipynb
- worked_examples/non-geospatial/classification_plant_phenotype/annotation_tasks_open.yaml
- worked_examples/postproc/postproc_compute_rail_density.ipynb

#### Removed

- tests/ignore_test_non_geo_pipeline.py

## [v1.0.5](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.5) (2023-07-25)

### Summary

The changelog includes several updates and modifications to the codebase, highlighted by the following key points:

1. **Error Management and Model Handling**:
   - An updated error message and the addition of functionality for saving the model state dictionary alongside the entire model.

2. **Testing Enhancements**:
   - New tests have been introduced for various models, include loading from strings and testing inference.
   - Tests have been expanded to include models from the Hugging Face library and TIMM.

3. **Dependency and Documentation Updates**:
   - Development dependencies for TIMM and transformers have been added.
   - Documentation, including the README and contribution guide, has been updated and Americanized, correcting typos and stylistic inconsistencies.

4. **File Handling Improvements**:
   - Transitioned from tab-separated to comma-separated default for files, allowing the use of `.tsv` extensions as well.

5. **Code Quality and Functional Fixes**:
   - Various error fixes and improvements to existing functionality, including a progress bar for download operations and adjustments to the Continuous Integration setup to better handle test uploads.

6. **New File Additions and Removals**:
   - Several new test files and examples have been added, while an outdated test file for the classifier has been removed.

Overall, the changes reflect a focus on improving functionality, enhancing testing coverage, and ensuring consistency within the code and its documentation.

### Commit messages

- update error message
- add saving of state_dict as well as whole model
- add tests for other models (load from string)
- add tests for hf and timm models
- fix errors
- Updates from download to annonate
- add tests for inference
- add dev dependencies (timm and transformers)
- add guidance for timm models
- add test
- add tqdm to sheet downloader
- make tqdm.auto throughout
- Update publish-to-test-pypi.yml to only run on review requested
- update docs
- change all files to comma separated as default
- fix problem with df_test=0
- allow .tsv files
- update tests
- add owslib to setup.py
- Update Contribution-guide.rst    add google form link to join slack workspace
- Update README.md
- fix typo
- americanize docs and readme
- americanize spelling in mapreader code
- americanize tests
- americanize worked examples
- fix broken annotate
- Unifies setup.py ".[dev]" install and CI "Install Tools" step
- Update About.rst
- Update test_annotations_loader.py

### Pull requests

- [andrewphilipsmith](https://github.com/Living-with-machines/MapReader/pull/222)
    > ### Summary
    > 
    > Describe the problem you're trying to fix in this pull request. 
    > Please reference any related issue and use fixes/close to automatically close them, if pertinent. For example: "Fixes #58", or "Addresses (but does not close) #238". -->
    > 
    > Fixes #<NUM>
    > 
    > ### Describe your changes
    >   
    > Describe changes/updates you have made to the code.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [ ] Self-review code
    > - [ ] Ensure submission passes current tests
    > - [ ] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/224)
    > ### Summary
    > 
    > Add guidance for timm models to classify.rst
    > Fixes #212 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/226)
    > ### Summary
    > 
    > Add progress bar to sheet downloader.
    > Fixes #194 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/225)
    > ### Summary
    > 
    > As per https://pytorch.org/tutorials/beginner/saving_loading_models.html, we should be saving the `state_dict()` of the model as opposed to (or in addition to) the entire model. 
    > This makes re-loading the model a bit more flexible and less likely to break.
    > 
    > This PR adds saving of the state_dict() into the classifier.save() method.
    > Addresses #187 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/227)
    > ### Summary
    > 
    > The test-pypi workflow seems to fail often. I think because its using the same file name (version) multiple times and then complains that the 'new' version its trying to upload already exists.
    > 
    > Potentially this could be fixed by only running once a PR is ready?
    > 
    > p.s. I think it accidentally made it fail on this PR as I requested your review twice. So obviously this doesn't completely fix the test-pypi issue.
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/244)
    > ### Summary
    > 
    > Currently we are using a mix of American and UK english.
    > This PR changes everything to American english.
    > 
    > Fixes #215 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/221)
    > ### Summary
    > 
    > Previously, all tests were with 'resnet18'.
    > This PR adds tests for non-resnet models, including add tests for loading models from hugging face and timm.
    > 
    > Fixes #214 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Add tests
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/241)
    > ### Summary
    > 
    > We currently save .csv files with "\t" as separator. I've changed this to "," so that they are actually .csv files.
    > I think this makes sense to do but would also be happy to switch to saving files as .tsv files.
    > 
    > Fixes #198 
    > 
    > ### Describe your changes
    > - change all cases of sep="\t" to sep=","
    > - ensure all save/load csv functions have argument to specify delimiter/sep
    > - update code in jupyter notebooks
    > - [x] update tests
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?

### Files

#### Added

- tests/sample_files/test_annots.tsv
- tests/sample_files/ts_downloaded_maps.tsv
- tests/test_classify/test_classifier.py
- worked_examples/geospatial/workshop_june_2023/annotation_tasks_coast.yaml

#### Modified

- .github/workflows/mr_ci.yml
- .github/workflows/mr_pip_ci.yml
- .github/workflows/publish-to-test-pypi.yml
- .gitignore
- LICENSE
- README.md
- docs/source/About.rst
- docs/source/Beginners-info.rst
- docs/source/Contribution-guide/Contribution-guide.rst
- docs/source/Contribution-guide/Documentation.rst
- docs/source/Contribution-guide/GitHub-guide.rst
- docs/source/Input-guidance.rst
- docs/source/Install.rst
- docs/source/User-guide/Annotate.rst
- docs/source/User-guide/Classify.rst
- docs/source/User-guide/Download.rst
- docs/source/User-guide/Load.rst
- docs/source/User-guide/User-guide.rst
- docs/source/Worked-examples/mnist_pipeline.ipynb
- docs/source/Worked-examples/one_inch_pipeline.ipynb
- docs/source/Worked-examples/plant_pipeline.ipynb
- mapreader/annotate/utils.py
- mapreader/classify/classifier.py
- mapreader/classify/classifier_context.py
- mapreader/classify/datasets.py
- mapreader/classify/load_annotations.py
- mapreader/download/downloader.py
- mapreader/download/downloader_utils.py
- mapreader/download/sheet_downloader.py
- mapreader/download/tile_loading.py
- mapreader/download/tile_merging.py
- mapreader/load/geo_utils.py
- mapreader/load/images.py
- mapreader/load/loader.py
- mapreader/utils/compute_and_save_stats.py
- mapreader/utils/slice_parallel.py
- setup.py
- tests/sample_files/test_annots.csv
- tests/sample_files/test_annots_append.csv
- tests/sample_files/ts_downloaded_maps.csv
- tests/test_import.py
- tests/test_load/test_geo_utils.py
- tests/test_load/test_images.py
- tests/test_load/test_images_load_parents.py
- tests/test_load/test_loader.py
- tests/test_load/test_loader_load_patches.py
- tests/test_sheet_downloader.py
- ways_of_working.md
- worked_examples/geospatial/classification_one_inch_maps/Pipeline.ipynb
- worked_examples/geospatial/workshop_june_2023/Workshop_June2023.ipynb
- worked_examples/geospatial/workshop_june_2023/Workshop_June2023_empty.ipynb
- worked_examples/non-geospatial/classification_mnist/Pipeline.ipynb
- worked_examples/non-geospatial/classification_plant_phenotype/Pipeline.ipynb

#### Removed

- tests/test_classifier.py

## [v1.0.4](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.4) (2023-06-27)

### Summary

The changelog includes two significant updates:

1. **Metadata Handling**: A new feature has been implemented to save metadata to a CSV file after each successful map image download. This change addresses issues with metadata not being created when downloads are canceled midway, thereby fixing related bugs (#196 and #205).

2. **Error Messages for Image Downloads**: An error message has been added to notify users when map images cannot be read due to partial downloads, advising them to redownload the affected images. This change addresses bug #204.

The updates were made in the `mapreader/download/sheet_downloader.py` and `mapreader/load/images.py` files. While checklist items for self-review and passing tests have been completed, additional tests are still pending.

### Commit messages

- save metadata to csv on each download
- add error message for broken image files|

### Pull requests

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/210)
    > ### Summary
    > 
    > Currently, if you stop a download halfway you can end up with partially created map images  which cannot be read by matplot.image.
    > 
    > This PR adds an error message if this is the case, and tells the user to go back and redownload that map image.
    > 
    > Fixes #204 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/209)
    > ### Summary
    > 
    > Currently, if you are downloading multiple maps and cancel your download part way through, the metadata.csv doesn't get created. 
    > This PR makes it so that the metadata.csv is updated after each image download and so should fix this issue..
    > 
    > Fixes #196 
    > Fixes #205
    > 
    > ### Describe your changes
    >   
    > - Add try/except so that download of map image is finished (not only if file exists).
    > - Save metadata after complete and succesful download of each map image.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [ ] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?

### Files

#### Modified

- mapreader/download/sheet_downloader.py
- mapreader/load/images.py

## [v1.0.3](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.3) (2023-06-21)

### Summary

The changelog indicates that minor details regarding version numbers have been added to the Developer documentation. This change is linked to a pull request by the user andrewphilipsmith. The commit includes a self-review process, but further tests and additions are pending. The file modified is `docs/source/Developers-guide.rst`.

### Commit messages

- adds details to dev docs about version numbers

### Pull requests

- [andrewphilipsmith](https://github.com/Living-with-machines/MapReader/pull/220)
    > ### Summary
    > 
    > Adds minor details to the Developer docs about version numbers.
    > 
    > Fixes #<NUM>
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [ ] Ensure submission passes current tests
    > - [ ] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?

### Files

### Modified

- docs/source/Developers-guide.rst

## [v1.0.2](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.2) (2023-06-20)

### Summary

The changelog outlines several updates and changes to the MapReader project, including:

#### Commit Messages:
- Updated documentation to include details on the NLS tile server.
- Transitioned GitHub workflows from a previous version named "alto2txt."
- Removed "poetry" as a build tool.
- Addressed a missing command-line switch.
- Improved version control history handling for `versioneer.py`.
- Altered the style of `versioneer.py`.
- Fixed issues related to deploying to production PyPI.
- Enabled manual triggers for workflows.

#### Pull Requests:
1. **PR #200** by andrewphilipsmith:
   - Introduces GitHub workflows for publishing to PyPI, where tagged commits deploy to the official PyPI and commits to the main branch deploy to Test PyPI.
   - Addresses issue #47.

2. **PR #197** by kmcdono2:
   - Provides proofreading and detailed explanations for fetching maps from tile server layers.
   - Aims to generalize the content beyond just the NLS context and improves documentation regarding tile layers, addressing issues #144 and partially #117.

3. **PR #202** by andrewphilipsmith:
   - Modifies the workflow for publishing tagged Python distributions to include a manual trigger for debugging purposes.

#### File Changes:
- **Added:** New GitHub workflow files for publishing to PyPI and Test PyPI.
- **Modified:** Various documentation and configuration files.
- **Removed:** No files were removed.

Overall, the updates focus on improving deployment workflows, enhancing documentation, and ensuring the new workflows operate correctly.

### Commit messages

- Update Input-guidance.rst w/NLS tile server details
- takes GH workflows from alto2txt
- removes poetry as build tool
- adds missing -m switch
- checkouts full git history for versioneer.py
- changes versioneer style
- fixes production PyPI deployment
- enables manual triggers

### Pull requests

- [andrewphilipsmith](https://github.com/Living-with-machines/MapReader/pull/200)
    > ### Summary
    > 
    > The PR adds Github workflows to publish to PyPI:
    > 
    > - `git tagged` commits get automatically deployed to https://pypi.org
    > - all commits to `main` branch get deployed to https://test.pypi.org
    > 
    > Addresses https://github.com/Living-with-machines/MapReader/issues/47
    > 
    > Fixes #<NUM>
    > 
    > ### Describe your changes
    >   
    > Adds two new GH workflows
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [ ] Ensure submission passes current tests
    > - [x] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [x] Everything looks ok?
- [kmcdono2](https://github.com/Living-with-machines/MapReader/pull/197)
    > ### Summary
    > 
    > Fixes #144 & partially #117 
    > 
    > ### Describe your changes
    >   
    > I've done a proofread, added comments to explain a bit more about what we do to get maps from tile server layers, and generally tried to explain things so they are less specific to NLS. I have also added the new, more detailed info about how to use any of the NLS tile layers.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review changes
    > 
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
    > - [ ] see note about how to explain geospatial coordinates we get from NLS and how these may vary in other collections. It would be nice to provide an example of something different. Not to resolve now, but additional comments there helpful for next step. Will open ticket about this.
    > - [ ] check amazon ws links for NLS metadata links are still active for examples provided in line
- [andrewphilipsmith](https://github.com/Living-with-machines/MapReader/pull/202)
    > ### Summary
    > 
    > The "Publish Tagged Python 🐍 distributions 📦 to PyPI" action is not triggering when expected.
    > This adds a manuual trigger to that workflow, for debug purposes.
    >   
    > - [ ] Self-review code
    > - [ ] Ensure submission passes current tests
    > - [ ] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?

### Files

#### Added

- .github/workflows/publish-to-pypi.yml
- .github/workflows/publish-to-test-pypi.yml

#### Modified

- .gitignore
- docs/source/Input-guidance.rst
- mapreader/_version.py
- setup.cfg

## [v1.0.1](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.1) (2023-06-06)

### Summary

Several updates and modifications related to a workshop notebook for geospatial work. Key contributions include:

- Initial creation of the workshop notebook and related resources.
- Fixes to bugs in the patchify function and image processing.
- Updates to the notebook based on feedback, including enhancements to the content and structure.
- Creation of an annotations file and an empty notebook.
- Addition of a new function `show_parent()` to the workbook.

Files added include the workshop notebook, an empty version of the notebook, an annotations YAML file, and a CSV with building annotations. Modifications were made to user documentation and a figure related to metadata plotting.

Overall, the updates focused on preparing and refining the workshop materials.

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

### Pull requests

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/188)
    > ### Summary
    > Creating the notebook for the workshop.

### Files

#### Added

- worked_examples/geospatial/workshop_june_2023/Workshop_June2023.ipynb
- worked_examples/geospatial/workshop_june_2023/Workshop_June2023_empty.ipynb
- worked_examples/geospatial/workshop_june_2023/annotation_tasks.yaml
- worked_examples/geospatial/workshop_june_2023/annotations/buildings_#workshop#.csv

#### Modified

- docs/source/User-guide/Download.rst
- docs/source/figures/plot_metadata_on_map.png

## [v1.0.0](https://github.com/Living-with-machines/MapReader/releases/tag/v1.0.0) (2023-06-06)

### Summary

The changelog details numerous updates and enhancements to the MapReader project. Key highlights include:

1. **Testing and Bug Fixes**: Multiple updates were made to tests, including renaming test files, fixing assertions, and enhancing error handling. Specific issues like coordinate assertion errors and PNG file loading problems were addressed.

2. **Code Improvements and Merges**: Significant refactoring was performed to align with the PEP8 style guide, remove obsolete methods, and improve image handling functions. Several feature branches were merged into the development branch, including enhancements for loader file paths and data inconsistencies.

3. **Feature Additions**: New functionality was introduced for handling metadata, including support for Excel files. The ability to save image patches in various formats (like GeoJSON and TIFF) was also added, alongside methods for calculating pixel statistics and handling coordinate data.

4. **Documentation Enhancements**: The developers' guide and contribution guidelines were expanded, and beginner information was included in the documentation. Various user guides were updated to reflect recent changes and improvements.

5. **General Maintenance**: Routine maintenance included minor typo fixes, updates to dependencies (such as `tqdm` for progress tracking), and reorganization of file structures for clarity.

6. **Pull Requests**: Several notable pull requests were merged, addressing issues related to metadata alignment, directory handling, and method improvements for downloading and processing map data.

Overall, the changelog reflects a concerted effort to improve the robustness, usability, and documentation of the MapReader package while adhering to best practices in coding and project management.

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

### Pull requests

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/95)
    > ### Summary
    > 
    > As per #82 - Language used to describe the different 'tasks' performed by the mapreader package do not always align between code/documentation/MR paper. This PR focuses on renaming/aligning language used throughout MapReader to ensure ease of use.
    > 
    > Addresses #82 
    > 
    > ### Describe your changes
    >   
    > - [x] rename loader (subpackage) to load
    > - [x] rename children to patches. 
    > - [x] rename slice/slicer/etc to patch/patchify/etc. Fixes #70 
    > - [ ] ~~rename TileServer (class) to TileServerClient~~ Since discussion leading to Issue no#99 we may need further updates to this TileServer class so am leaving for now
    > - [x] rename train to learn **NOTE: I have done this but may be better to split it into training and inference** 
    > - [x] Change default path_save for patches from 'test' to 'patches'
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/140)
    > ### Summary
    > At the moment, loader() method requires `path/to/files/*.png` to find files but no error is returned if you pass `path/to/directory` or `path/to/directory/` to loader() method. 
    > This should be fixed to a) allow path to directory and b) return errors in outside cases.
    > 
    > Fixes #110 
    > 
    > ### Describe your changes
    > 
    > - [x] If path to directory is given, get files within directory
    > - [x] Add `file_ext` argument which can be specified if path to directory is given
    > - [x] Return errors if files of multiple file extensions are found in path_images
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add doc strings and update docs
    > - [x] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [kallewesterling](https://github.com/Living-with-machines/MapReader/pull/155)
    > Adding my name and contact information to the current project members.
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/151)
    > ### Summary
    > Loader should return warnings if your metadata and images do not align.
    > Fixes #116 
    > 
    > ### Describe your changes
    > 
    > Rewritten add_metadata function, specifically:
    > - metadata_df now only contains ``columns`` (if specified)
    > - if you pass path to ``metadata_file`` but without ``.csv`` on the end, append ``.csv`` to your path. This is thinking particularly about windows users and file explorer not showing file extensions.
    > - print warnings for missing and extra metadata
    > - try eval(item) for all metadata items (allows for typos such as "polygone" in column headers)
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/165)
    > Add code of conduct (sorry had to make a new PR as previous PR was to merge into ``dev`` branch).
    > 
    > - @kmcdono2 can you just double check you are happy with the contents of this COC?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/160)
    > ### Summary
    > Add error message for 32-bit (or other not-implemented) image modes. 
    > Fixes #131 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/163)
    > ### Summary
    > Updates to load package. 
    > 
    > Fixes #<NUM>
    > 
    > ### Describe your changes
    > - Rename classes/methods/functions to align with PEP 8 (see ticket #152 )
    > - Create ``self.parents`` and ``self.patches`` attributes for ``MapImages`` object
    > - Auto add shape and try add geo_info when initialising patches
    > - Add patch to parent if tree_level = 'patch' when running ``image_constructor``
    > - "coord" is now "coordinates" and are "xmin, ymin, xmax, ymax"
    > - Save patch coordinates when slicing (see ticket #138 )
    > - Remove ``slicer.py`` and instead just have slicer as part of ``MapImages`` class
    > - If no ``path_save`` is passed, patches are now saved in f"patches_{patch_size}_{method}" directory (see ticket #139 )
    > - Patch pixel bounds are now saved as ``pixel_bounds`` tuple (xmin, ymin, xmax, ymax).
    > - Pixel stats now calculated for each channel in image (auto pick out channels instead of assume RGB).
    > - Redo/simplify plotting **RW: to discuss**
    > - Split ``add_geo_info`` into function which calls ``add_geo_info_id`` for each individual image_id in ``self.patches``.
    > - add method to get tree_level from image_id (``get_tree_level()``) - no longer need to specify tree_level if specifying image_id argument, removes possibility of conflicting tree_level and image_id.
    > - Update ``tests/sample_files`` to have 9x9 pixel images instead of full size images
    > - add ``geo_utils`` to load subpackage - fixes #97 
    > - Add progress bars for functions (e.g. patchify, load, plotting).
    > - Explicitly print info for where patches are saved - addresses #122 
    > - Allow metadata as excel file as well as csv.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [ ] Self-review code
    > - [ ] Ensure submission passes current tests
    > - [ ] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/180)

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/154)
    > ### Summary
    > 
    > Updates to the downloader subpackage to allow a variety of different methods for downloading map sheets.
    > Fixes #108 
    > Fixes #99 
    > 
    > ### Describe your changes
    > Two classes when downloading maps now:
    > 1) SheetDownloader - for if you have metadata and want to download and merge tiles from a tileserver which correspond to each map sheet
    > 2) Downloader - for if you don't have metadata but want to download and merge tiles from a tileserver using a polygon (area) to define each 'map image'
    > 
    > SheetDownloader contains methods for downloading :
    > - All map sheets in metadata
    > - Map sheets overlapping with a defined polygon (area)
    > - Map sheets completely within a defined polygon (area)
    > - By WFS index no. (see pic below)
    > - Map sheets that contain a single point (coordinate)
    > - Using query results list (queries can be any of the above methods but are essentially done without downloading)
    > 
    > Download code now taken from https://github.com/baurls/TileStitcher.
    > (Maybe can bring in some of old code to improve download speed but will need to look into this a bit more - RW).
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/176)
    > ### Summary
    > 
    > Have added a simple method of saving your parent and patch dataframes (i.e. all information in you MapImages instance) to the ``convert_images()`` method. 
    > Fixes #174 
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/164)
    > ### Summary
    > 
    > Updates to docs
    > Begins to address #159 
    > Fixes #177 
    > 
    > ### Describe your changes
    > - add "beginners info"
    > - split contribution guide into 3 sections/types of contributors (plus a "GitHub guide"): tutorials, documentation and code
    > - update load user guide
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Check on RTD
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/182)
    > ### Summary
    > Allow patches to be saved as geojson.
    > Fixes #37  
    > 
    > ### Describe your changes
    > Added ``add_patch_polgyons`` and  ``save_patches_as_geojson`` methods to ``MapImages`` object.
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/185)
    > ### Summary
    > General updates to train/learn subpackage.
    > Fixes #186 
    > 
    > ### Describe your changes
    > - `AnnotationsLoader` is now part of this subpackage and now creates the datasets and dataloaders
    > - `ClassifierContainer` MUST be loaded with a model, dataloaders and labels_map
    > - PEP8 style guide updates
    > - [x] Update context parts
    > - [x] Allow hugging face models (transformers) - #186 
    > - [x] Update docs
    > - [x] Rename to ``classify``
    > 
    > ### Checklist before assigning a reviewer (update as needed)
    >   
    > - [x] Self-review code
    > - [x] Ensure submission passes current tests
    > - [x] Add tests
    >   
    > ### Reviewer checklist
    >   
    > Please add anything you want reviewers to specifically focus/comment on.
    > 
    > - [ ] Everything looks ok?
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/199)

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/195)
    > Require python versions 3.7 - 3.10. 

### Files

#### Added

- .github/ISSUE_TEMPLATE/documentation_update.md
- .github/PULL_REQUEST_TEMPLATE.md
- .github/workflows/publish-to-conda-forge.yml
- CODE_OF_CONDUCT.md
- docs/source/About.rst
- docs/source/Beginners-info.rst
- docs/source/Contribution-guide/Code.rst
- docs/source/Contribution-guide/Contribution-guide.rst
- docs/source/Contribution-guide/Documentation.rst
- docs/source/Contribution-guide/GitHub-guide.rst
- docs/source/Contribution-guide/Worked-examples.rst
- docs/source/Developers-guide.rst
- docs/source/User-guide/Classify.rst
- docs/source/Worked-examples/Worked-examples.rst
- docs/source/Worked-examples/mnist_pipeline.ipynb
- docs/source/Worked-examples/one_inch_pipeline.ipynb
- docs/source/Worked-examples/plant_pipeline.ipynb
- docs/source/figures/in_out_annotate.png
- docs/source/figures/in_out_classify.png
- docs/source/figures/in_out_download.png
- docs/source/figures/in_out_load.png

#### Modified

- .github/ISSUE_TEMPLATE/bug_report.md
- .github/ISSUE_TEMPLATE/feature_request.md
- .gitignore
- README.md
- conda/meta.yaml
- docs/requirements.txt
- docs/source/Input-guidance.rst
- docs/source/Install.rst
- docs/source/User-guide/Annotate.rst
- docs/source/User-guide/Download.rst
- docs/source/User-guide/Load.rst
- docs/source/User-guide/User-guide.rst
- docs/source/api/index.rst
- docs/source/conf.py
- docs/source/figures/annotate.png
- docs/source/figures/annotate_context.png

#### Removed

- docs/build/doctrees/API-reference.doctree
- docs/build/doctrees/Annotate.doctree
- docs/build/doctrees/Download.doctree
- docs/build/doctrees/Geospatial-examples.doctree
- docs/build/doctrees/Input-guidance.doctree
- docs/build/doctrees/Install.doctree
- docs/build/doctrees/Load.doctree
- docs/build/doctrees/Non-geospatial-examples.doctree
- docs/build/doctrees/Patchify.doctree
- docs/build/doctrees/Post-process.doctree
- docs/build/doctrees/Process.doctree
- docs/build/doctrees/README.doctree
- docs/build/doctrees/Train.doctree
- docs/build/doctrees/User-guide.doctree
- docs/build/doctrees/User-guide/Annotate.doctree
- docs/build/doctrees/User-guide/Download.doctree
- docs/build/doctrees/User-guide/Geospatial-examples.doctree
- docs/build/doctrees/User-guide/Load.doctree
- docs/build/doctrees/User-guide/Non-geospatial-examples.doctree
- docs/build/doctrees/User-guide/Patchify.doctree
- docs/build/doctrees/User-guide/Post-process.doctree
- docs/build/doctrees/User-guide/Process.doctree
- docs/build/doctrees/User-guide/Train.doctree
- docs/build/doctrees/User-guide/User-guide.doctree
- docs/build/doctrees/User-guide/Worked-examples.doctree
- docs/build/doctrees/User-guide/index.doctree
- docs/build/doctrees/Worked-examples.doctree
- docs/build/doctrees/api/index.doctree
- docs/build/doctrees/api/mapreader/annotate/index.doctree
- docs/build/doctrees/api/mapreader/annotate/load_annotate/index.doctree
- docs/build/doctrees/api/mapreader/annotate/utils/index.doctree
- docs/build/doctrees/api/mapreader/download/azure_access/index.doctree
- docs/build/doctrees/api/mapreader/download/index.doctree
- docs/build/doctrees/api/mapreader/download/tileserver_access/index.doctree
- docs/build/doctrees/api/mapreader/download/tileserver_helpers/index.doctree
- docs/build/doctrees/api/mapreader/download/tileserver_scraper/index.doctree
- docs/build/doctrees/api/mapreader/download/tileserver_stitcher/index.doctree
- docs/build/doctrees/api/mapreader/index.doctree
- docs/build/doctrees/api/mapreader/loader/images/index.doctree
- docs/build/doctrees/api/mapreader/loader/index.doctree
- docs/build/doctrees/api/mapreader/loader/loader/index.doctree
- docs/build/doctrees/api/mapreader/process/index.doctree
- docs/build/doctrees/api/mapreader/process/process/index.doctree
- docs/build/doctrees/api/mapreader/slicers/index.doctree
- docs/build/doctrees/api/mapreader/slicers/slicers/index.doctree
- docs/build/doctrees/api/mapreader/train/classifier/index.doctree
- docs/build/doctrees/api/mapreader/train/classifier_context/index.doctree
- docs/build/doctrees/api/mapreader/train/custom_models/index.doctree
- docs/build/doctrees/api/mapreader/train/datasets/index.doctree
- docs/build/doctrees/api/mapreader/train/index.doctree
- docs/build/doctrees/api/mapreader/utils/compute_and_save_stats/index.doctree
- docs/build/doctrees/api/mapreader/utils/index.doctree
- docs/build/doctrees/api/mapreader/utils/slice_parallel/index.doctree
- docs/build/doctrees/api/mapreader/utils/utils/index.doctree
- docs/build/doctrees/environment.pickle
- docs/build/doctrees/index.doctree
- docs/build/html/.buildinfo
- docs/build/html/API-reference.html
- docs/build/html/Annotate.html
- docs/build/html/Download.html
- docs/build/html/Geospatial-examples.html
- docs/build/html/Input-guidance.html
- docs/build/html/Install.html
- docs/build/html/Load.html
- docs/build/html/Non-geospatial-examples.html
- docs/build/html/Patchify.html
- docs/build/html/Post-process.html
- docs/build/html/Process.html
- docs/build/html/README.html
- docs/build/html/Train.html
- docs/build/html/User-guide.html
- docs/build/html/User-guide/Annotate.html
- docs/build/html/User-guide/Download.html
- docs/build/html/User-guide/Geospatial-examples.html
- docs/build/html/User-guide/Load.html
- docs/build/html/User-guide/Non-geospatial-examples.html
- docs/build/html/User-guide/Patchify.html
- docs/build/html/User-guide/Post-process.html
- docs/build/html/User-guide/Process.html
- docs/build/html/User-guide/Train.html
- docs/build/html/User-guide/User-guide.html
- docs/build/html/User-guide/Worked-examples.html
- docs/build/html/User-guide/index.html
- docs/build/html/Worked-examples.html
- docs/build/html/_images/annotate.png
- docs/build/html/_images/annotate_context.png
- docs/build/html/_images/hist_published_dates.png
- docs/build/html/_images/loss.png
- docs/build/html/_images/plot_metadata_on_map.png
- docs/build/html/_images/show_image.png
- docs/build/html/_images/show_image_labels_10.png
- docs/build/html/_images/show_par.png
- docs/build/html/_images/show_par_RGB.png
- docs/build/html/_images/show_par_RGB_0.5.png
- docs/build/html/_images/show_sample_child.png
- docs/build/html/_images/show_sample_parent.png
- docs/build/html/_images/show_sample_train_8.png
- docs/build/html/_images/show_sample_val_8.png
- docs/build/html/_sources/API-reference.rst.txt
- docs/build/html/_sources/Annotate.rst.txt
- docs/build/html/_sources/Download.rst.txt
- docs/build/html/_sources/Geospatial-examples.rst.txt
- docs/build/html/_sources/Input-guidance.rst.txt
- docs/build/html/_sources/Install.rst.txt
- docs/build/html/_sources/Load.rst.txt
- docs/build/html/_sources/Non-geospatial-examples.rst.txt
- docs/build/html/_sources/Patchify.rst.txt
- docs/build/html/_sources/Post-process.rst.txt
- docs/build/html/_sources/Process.rst.txt
- docs/build/html/_sources/README.rst.txt
- docs/build/html/_sources/Train.rst.txt
- docs/build/html/_sources/User-guide.rst.txt
- docs/build/html/_sources/User-guide/Annotate.rst.txt
- docs/build/html/_sources/User-guide/Download.rst.txt
- docs/build/html/_sources/User-guide/Geospatial-examples.rst.txt
- docs/build/html/_sources/User-guide/Load.rst.txt
- docs/build/html/_sources/User-guide/Non-geospatial-examples.rst.txt
- docs/build/html/_sources/User-guide/Patchify.rst.txt
- docs/build/html/_sources/User-guide/Post-process.rst.txt
- docs/build/html/_sources/User-guide/Process.rst.txt
- docs/build/html/_sources/User-guide/Train.rst.txt
- docs/build/html/_sources/User-guide/User-guide.rst.txt
- docs/build/html/_sources/User-guide/Worked-examples.rst.txt
- docs/build/html/_sources/User-guide/index.rst.txt
- docs/build/html/_sources/Worked-examples.rst.txt
- docs/build/html/_sources/api/index.rst.txt
- docs/build/html/_sources/api/mapreader/annotate/index.rst.txt
- docs/build/html/_sources/api/mapreader/annotate/load_annotate/index.rst.txt
- docs/build/html/_sources/api/mapreader/annotate/utils/index.rst.txt
- docs/build/html/_sources/api/mapreader/download/azure_access/index.rst.txt
- docs/build/html/_sources/api/mapreader/download/index.rst.txt
- docs/build/html/_sources/api/mapreader/download/tileserver_access/index.rst.txt
- docs/build/html/_sources/api/mapreader/download/tileserver_helpers/index.rst.txt
- docs/build/html/_sources/api/mapreader/download/tileserver_scraper/index.rst.txt
- docs/build/html/_sources/api/mapreader/download/tileserver_stitcher/index.rst.txt
- docs/build/html/_sources/api/mapreader/index.rst.txt
- docs/build/html/_sources/api/mapreader/loader/images/index.rst.txt
- docs/build/html/_sources/api/mapreader/loader/index.rst.txt
- docs/build/html/_sources/api/mapreader/loader/loader/index.rst.txt
- docs/build/html/_sources/api/mapreader/process/index.rst.txt
- docs/build/html/_sources/api/mapreader/process/process/index.rst.txt
- docs/build/html/_sources/api/mapreader/slicers/index.rst.txt
- docs/build/html/_sources/api/mapreader/slicers/slicers/index.rst.txt
- docs/build/html/_sources/api/mapreader/train/classifier/index.rst.txt
- docs/build/html/_sources/api/mapreader/train/classifier_context/index.rst.txt
- docs/build/html/_sources/api/mapreader/train/custom_models/index.rst.txt
- docs/build/html/_sources/api/mapreader/train/datasets/index.rst.txt
- docs/build/html/_sources/api/mapreader/train/index.rst.txt
- docs/build/html/_sources/api/mapreader/utils/compute_and_save_stats/index.rst.txt
- docs/build/html/_sources/api/mapreader/utils/index.rst.txt
- docs/build/html/_sources/api/mapreader/utils/slice_parallel/index.rst.txt
- docs/build/html/_sources/api/mapreader/utils/utils/index.rst.txt
- docs/build/html/_sources/index.rst.txt
- docs/build/html/_static/_sphinx_javascript_frameworks_compat.js
- docs/build/html/_static/basic.css
- docs/build/html/_static/check-solid.svg
- docs/build/html/_static/clipboard.min.js
- docs/build/html/_static/copy-button.svg
- docs/build/html/_static/copybutton.css
- docs/build/html/_static/copybutton.js
- docs/build/html/_static/copybutton_funcs.js
- docs/build/html/_static/css/badge_only.css
- docs/build/html/_static/css/fonts/Roboto-Slab-Bold.woff
- docs/build/html/_static/css/fonts/Roboto-Slab-Bold.woff2
- docs/build/html/_static/css/fonts/Roboto-Slab-Regular.woff
- docs/build/html/_static/css/fonts/Roboto-Slab-Regular.woff2
- docs/build/html/_static/css/fonts/fontawesome-webfont.eot
- docs/build/html/_static/css/fonts/fontawesome-webfont.svg
- docs/build/html/_static/css/fonts/fontawesome-webfont.ttf
- docs/build/html/_static/css/fonts/fontawesome-webfont.woff
- docs/build/html/_static/css/fonts/fontawesome-webfont.woff2
- docs/build/html/_static/css/fonts/lato-bold-italic.woff
- docs/build/html/_static/css/fonts/lato-bold-italic.woff2
- docs/build/html/_static/css/fonts/lato-bold.woff
- docs/build/html/_static/css/fonts/lato-bold.woff2
- docs/build/html/_static/css/fonts/lato-normal-italic.woff
- docs/build/html/_static/css/fonts/lato-normal-italic.woff2
- docs/build/html/_static/css/fonts/lato-normal.woff
- docs/build/html/_static/css/fonts/lato-normal.woff2
- docs/build/html/_static/css/theme.css
- docs/build/html/_static/disqus.js
- docs/build/html/_static/doctools.js
- docs/build/html/_static/documentation_options.js
- docs/build/html/_static/file.png
- docs/build/html/_static/graphviz.css
- docs/build/html/_static/jquery-3.6.0.js
- docs/build/html/_static/jquery.js
- docs/build/html/_static/js/badge_only.js
- docs/build/html/_static/js/html5shiv-printshiv.min.js
- docs/build/html/_static/js/html5shiv.min.js
- docs/build/html/_static/js/theme.js
- docs/build/html/_static/language_data.js
- docs/build/html/_static/minus.png
- docs/build/html/_static/plus.png
- docs/build/html/_static/pygments.css
- docs/build/html/_static/searchtools.js
- docs/build/html/_static/sphinx_highlight.js
- docs/build/html/_static/underscore-1.13.1.js
- docs/build/html/_static/underscore.js
- docs/build/html/api/index.html
- docs/build/html/api/mapreader/annotate/index.html
- docs/build/html/api/mapreader/annotate/load_annotate/index.html
- docs/build/html/api/mapreader/annotate/utils/index.html
- docs/build/html/api/mapreader/download/azure_access/index.html
- docs/build/html/api/mapreader/download/index.html
- docs/build/html/api/mapreader/download/tileserver_access/index.html
- docs/build/html/api/mapreader/download/tileserver_helpers/index.html
- docs/build/html/api/mapreader/download/tileserver_scraper/index.html
- docs/build/html/api/mapreader/download/tileserver_stitcher/index.html
- docs/build/html/api/mapreader/index.html
- docs/build/html/api/mapreader/loader/images/index.html
- docs/build/html/api/mapreader/loader/index.html
- docs/build/html/api/mapreader/loader/loader/index.html
- docs/build/html/api/mapreader/process/index.html
- docs/build/html/api/mapreader/process/process/index.html
- docs/build/html/api/mapreader/slicers/index.html
- docs/build/html/api/mapreader/slicers/slicers/index.html
- docs/build/html/api/mapreader/train/classifier/index.html
- docs/build/html/api/mapreader/train/classifier_context/index.html
- docs/build/html/api/mapreader/train/custom_models/index.html
- docs/build/html/api/mapreader/train/datasets/index.html
- docs/build/html/api/mapreader/train/index.html
- docs/build/html/api/mapreader/utils/compute_and_save_stats/index.html
- docs/build/html/api/mapreader/utils/index.html
- docs/build/html/api/mapreader/utils/slice_parallel/index.html
- docs/build/html/api/mapreader/utils/utils/index.html
- docs/build/html/genindex.html
- docs/build/html/index.html
- docs/build/html/objects.inv
- docs/build/html/py-modindex.html
- docs/build/html/search.html
- docs/build/html/searchindex.js
- docs/source/Developers-guide/Developers-guide.rst
- docs/source/User-guide/Geospatial-examples.rst
- docs/source/User-guide/Non-geospatial-examples.rst
- docs/source/User-guide/Train.rst
- docs/source/User-guide/Worked-examples.rst
- docs/source/api/mapreader/annotate/index.rst
- docs/source/api/mapreader/annotate/load_annotate/index.rst
- docs/source/api/mapreader/annotate/utils/index.rst
- docs/source/api/mapreader/download/azure_access/index.rst
- docs/source/api/mapreader/download/index.rst
- docs/source/api/mapreader/download/tileserver_access/index.rst
- docs/source/api/mapreader/download/tileserver_helpers/index.rst
- docs/source/api/mapreader/download/tileserver_scraper/index.rst
- docs/source/api/mapreader/download/tileserver_stitcher/index.rst
- docs/source/api/mapreader/index.rst
- docs/source/api/mapreader/loader/images/index.rst
- docs/source/api/mapreader/loader/index.rst
- docs/source/api/mapreader/loader/loader/index.rst
- docs/source/api/mapreader/process/index.rst
- docs/source/api/mapreader/process/process/index.rst
- docs/source/api/mapreader/slicers/index.rst
- docs/source/api/mapreader/slicers/slicers/index.rst
- docs/source/api/mapreader/train/classifier/index.rst
- docs/source/api/mapreader/train/classifier_context/index.rst
- docs/source/api/mapreader/train/custom_models/index.rst
- docs/source/api/mapreader/train/datasets/index.rst
- docs/source/api/mapreader/train/index.rst
- docs/source/api/mapreader/utils/compute_and_save_stats/index.rst
- docs/source/api/mapreader/utils/index.rst
- docs/source/api/mapreader/utils/slice_parallel/index.rst
- docs/source/api/mapreader/utils/utils/index.rst

## [v0.3.4](https://github.com/Living-with-machines/MapReader/releases/tag/v0.3.4) (2023-02-20)

### Summary

**New Features:**
- Introduced Continuous Integration (CI) for pip installation.
- Added issue templates for bug reporting and feature requests.
- Documented processes with new contribution guidelines and additional details in the documentation.
- Included `versioneer` for version management in the project.

**Documentation Updates:**
- Links and citations were updated from ArXiv to the ACM paper.
- Updates made to the `README.md` and several documentation files (e.g., `Load.rst`, `Train.rst`) based on plant tutorial insights.
- Basic documentation added for downloading, loading, annotating, and training.
- Organised documentation files into directories by section and improved documentation structure.

**File Management:**
- Added essential files like `.gitattributes`, `.readthedocs.yaml`, `requirements.txt`, and various Conda configuration files.
- Several `.rst` and HTML files were generated during the documentation build process, including API references and user guides.

**Modifications:**
- The README was modified for clarity and content improvements.

**Fixes:**
- Corrected issues in `requirements.txt` and fixed images in `Load.rst`.
- Streamlined Conda dependencies to enhance resolution.

**Miscellaneous:**
- Several initial commits highlighted the foundational setup of the project.
- Successful local builds of conda packages occurred, ensuring the deployment process was functioning correctly. 

Overall, this release focused on improving documentation, incorporating CI, and resolving package management issues.

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

### Pull requests

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/65)

- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/68)
    > Added in extra details to align with plant tutorial:
    > 
    > 1. show_par()
    > 2. calc_pixel_stats()
    > 3. convertImages()
    > 
- [rwood-97](https://github.com/Living-with-machines/MapReader/pull/69)

### Files

#### Added

- .gitattributes
- .github/ISSUE_TEMPLATE/bug_report.md
- .github/ISSUE_TEMPLATE/feature_request.md
- .github/workflows/mr_pip_ci.yml
- .readthedocs.yaml
- conda/ipyannotate/conda/meta.yaml
- conda/ipyannotate/conda/setup.py
- conda/meta.yaml
- conda/parhugin/conda/meta.yaml
- docs/.contents.rst.swp
- docs/Makefile
- docs/build/doctrees/API-reference.doctree
- docs/build/doctrees/Annotate.doctree
- docs/build/doctrees/Download.doctree
- docs/build/doctrees/Geospatial-examples.doctree
- docs/build/doctrees/Input-guidance.doctree
- docs/build/doctrees/Install.doctree
- docs/build/doctrees/Load.doctree
- docs/build/doctrees/Non-geospatial-examples.doctree
- docs/build/doctrees/Patchify.doctree
- docs/build/doctrees/Post-process.doctree
- docs/build/doctrees/Process.doctree
- docs/build/doctrees/README.doctree
- docs/build/doctrees/Train.doctree
- docs/build/doctrees/User-guide.doctree
- docs/build/doctrees/User-guide/Annotate.doctree
- docs/build/doctrees/User-guide/Download.doctree
- docs/build/doctrees/User-guide/Geospatial-examples.doctree
- docs/build/doctrees/User-guide/Load.doctree
- docs/build/doctrees/User-guide/Non-geospatial-examples.doctree
- docs/build/doctrees/User-guide/Patchify.doctree
- docs/build/doctrees/User-guide/Post-process.doctree
- docs/build/doctrees/User-guide/Process.doctree
- docs/build/doctrees/User-guide/Train.doctree
- docs/build/doctrees/User-guide/User-guide.doctree
- docs/build/doctrees/User-guide/Worked-examples.doctree
- docs/build/doctrees/User-guide/index.doctree
- docs/build/doctrees/Worked-examples.doctree
- docs/build/doctrees/api/index.doctree
- docs/build/doctrees/api/mapreader/annotate/index.doctree
- docs/build/doctrees/api/mapreader/annotate/load_annotate/index.doctree
- docs/build/doctrees/api/mapreader/annotate/utils/index.doctree
- docs/build/doctrees/api/mapreader/download/azure_access/index.doctree
- docs/build/doctrees/api/mapreader/download/index.doctree
- docs/build/doctrees/api/mapreader/download/tileserver_access/index.doctree
- docs/build/doctrees/api/mapreader/download/tileserver_helpers/index.doctree
- docs/build/doctrees/api/mapreader/download/tileserver_scraper/index.doctree
- docs/build/doctrees/api/mapreader/download/tileserver_stitcher/index.doctree
- docs/build/doctrees/api/mapreader/index.doctree
- docs/build/doctrees/api/mapreader/loader/images/index.doctree
- docs/build/doctrees/api/mapreader/loader/index.doctree
- docs/build/doctrees/api/mapreader/loader/loader/index.doctree
- docs/build/doctrees/api/mapreader/process/index.doctree
- docs/build/doctrees/api/mapreader/process/process/index.doctree
- docs/build/doctrees/api/mapreader/slicers/index.doctree
- docs/build/doctrees/api/mapreader/slicers/slicers/index.doctree
- docs/build/doctrees/api/mapreader/train/classifier/index.doctree
- docs/build/doctrees/api/mapreader/train/classifier_context/index.doctree
- docs/build/doctrees/api/mapreader/train/custom_models/index.doctree
- docs/build/doctrees/api/mapreader/train/datasets/index.doctree
- docs/build/doctrees/api/mapreader/train/index.doctree
- docs/build/doctrees/api/mapreader/utils/compute_and_save_stats/index.doctree
- docs/build/doctrees/api/mapreader/utils/index.doctree
- docs/build/doctrees/api/mapreader/utils/slice_parallel/index.doctree
- docs/build/doctrees/api/mapreader/utils/utils/index.doctree
- docs/build/doctrees/environment.pickle
- docs/build/doctrees/index.doctree
- docs/build/html/.buildinfo
- docs/build/html/API-reference.html
- docs/build/html/Annotate.html
- docs/build/html/Download.html
- docs/build/html/Geospatial-examples.html
- docs/build/html/Input-guidance.html
- docs/build/html/Install.html
- docs/build/html/Load.html
- docs/build/html/Non-geospatial-examples.html
- docs/build/html/Patchify.html
- docs/build/html/Post-process.html
- docs/build/html/Process.html
- docs/build/html/README.html
- docs/build/html/Train.html
- docs/build/html/User-guide.html
- docs/build/html/User-guide/Annotate.html
- docs/build/html/User-guide/Download.html
- docs/build/html/User-guide/Geospatial-examples.html
- docs/build/html/User-guide/Load.html
- docs/build/html/User-guide/Non-geospatial-examples.html
- docs/build/html/User-guide/Patchify.html
- docs/build/html/User-guide/Post-process.html
- docs/build/html/User-guide/Process.html
- docs/build/html/User-guide/Train.html
- docs/build/html/User-guide/User-guide.html
- docs/build/html/User-guide/Worked-examples.html
- docs/build/html/User-guide/index.html
- docs/build/html/Worked-examples.html
- docs/build/html/_images/annotate.png
- docs/build/html/_images/annotate_context.png
- docs/build/html/_images/hist_published_dates.png
- docs/build/html/_images/loss.png
- docs/build/html/_images/plot_metadata_on_map.png
- docs/build/html/_images/show_image.png
- docs/build/html/_images/show_image_labels_10.png
- docs/build/html/_images/show_par.png
- docs/build/html/_images/show_par_RGB.png
- docs/build/html/_images/show_par_RGB_0.5.png
- docs/build/html/_images/show_sample_child.png
- docs/build/html/_images/show_sample_parent.png
- docs/build/html/_images/show_sample_train_8.png
- docs/build/html/_images/show_sample_val_8.png
- docs/build/html/_sources/API-reference.rst.txt
- docs/build/html/_sources/Annotate.rst.txt
- docs/build/html/_sources/Download.rst.txt
- docs/build/html/_sources/Geospatial-examples.rst.txt
- docs/build/html/_sources/Input-guidance.rst.txt
- docs/build/html/_sources/Install.rst.txt
- docs/build/html/_sources/Load.rst.txt
- docs/build/html/_sources/Non-geospatial-examples.rst.txt
- docs/build/html/_sources/Patchify.rst.txt
- docs/build/html/_sources/Post-process.rst.txt
- docs/build/html/_sources/Process.rst.txt
- docs/build/html/_sources/README.rst.txt
- docs/build/html/_sources/Train.rst.txt
- docs/build/html/_sources/User-guide.rst.txt
- docs/build/html/_sources/User-guide/Annotate.rst.txt
- docs/build/html/_sources/User-guide/Download.rst.txt
- docs/build/html/_sources/User-guide/Geospatial-examples.rst.txt
- docs/build/html/_sources/User-guide/Load.rst.txt
- docs/build/html/_sources/User-guide/Non-geospatial-examples.rst.txt
- docs/build/html/_sources/User-guide/Patchify.rst.txt
- docs/build/html/_sources/User-guide/Post-process.rst.txt
- docs/build/html/_sources/User-guide/Process.rst.txt
- docs/build/html/_sources/User-guide/Train.rst.txt
- docs/build/html/_sources/User-guide/User-guide.rst.txt
- docs/build/html/_sources/User-guide/Worked-examples.rst.txt
- docs/build/html/_sources/User-guide/index.rst.txt
- docs/build/html/_sources/Worked-examples.rst.txt
- docs/build/html/_sources/api/index.rst.txt
- docs/build/html/_sources/api/mapreader/annotate/index.rst.txt
- docs/build/html/_sources/api/mapreader/annotate/load_annotate/index.rst.txt
- docs/build/html/_sources/api/mapreader/annotate/utils/index.rst.txt
- docs/build/html/_sources/api/mapreader/download/azure_access/index.rst.txt
- docs/build/html/_sources/api/mapreader/download/index.rst.txt
- docs/build/html/_sources/api/mapreader/download/tileserver_access/index.rst.txt
- docs/build/html/_sources/api/mapreader/download/tileserver_helpers/index.rst.txt
- docs/build/html/_sources/api/mapreader/download/tileserver_scraper/index.rst.txt
- docs/build/html/_sources/api/mapreader/download/tileserver_stitcher/index.rst.txt
- docs/build/html/_sources/api/mapreader/index.rst.txt
- docs/build/html/_sources/api/mapreader/loader/images/index.rst.txt
- docs/build/html/_sources/api/mapreader/loader/index.rst.txt
- docs/build/html/_sources/api/mapreader/loader/loader/index.rst.txt
- docs/build/html/_sources/api/mapreader/process/index.rst.txt
- docs/build/html/_sources/api/mapreader/process/process/index.rst.txt
- docs/build/html/_sources/api/mapreader/slicers/index.rst.txt
- docs/build/html/_sources/api/mapreader/slicers/slicers/index.rst.txt
- docs/build/html/_sources/api/mapreader/train/classifier/index.rst.txt
- docs/build/html/_sources/api/mapreader/train/classifier_context/index.rst.txt
- docs/build/html/_sources/api/mapreader/train/custom_models/index.rst.txt
- docs/build/html/_sources/api/mapreader/train/datasets/index.rst.txt
- docs/build/html/_sources/api/mapreader/train/index.rst.txt
- docs/build/html/_sources/api/mapreader/utils/compute_and_save_stats/index.rst.txt
- docs/build/html/_sources/api/mapreader/utils/index.rst.txt
- docs/build/html/_sources/api/mapreader/utils/slice_parallel/index.rst.txt
- docs/build/html/_sources/api/mapreader/utils/utils/index.rst.txt
- docs/build/html/_sources/index.rst.txt
- docs/build/html/_static/_sphinx_javascript_frameworks_compat.js
- docs/build/html/_static/basic.css
- docs/build/html/_static/check-solid.svg
- docs/build/html/_static/clipboard.min.js
- docs/build/html/_static/copy-button.svg
- docs/build/html/_static/copybutton.css
- docs/build/html/_static/copybutton.js
- docs/build/html/_static/copybutton_funcs.js
- docs/build/html/_static/css/badge_only.css
- docs/build/html/_static/css/fonts/Roboto-Slab-Bold.woff
- docs/build/html/_static/css/fonts/Roboto-Slab-Bold.woff2
- docs/build/html/_static/css/fonts/Roboto-Slab-Regular.woff
- docs/build/html/_static/css/fonts/Roboto-Slab-Regular.woff2
- docs/build/html/_static/css/fonts/fontawesome-webfont.eot
- docs/build/html/_static/css/fonts/fontawesome-webfont.svg
- docs/build/html/_static/css/fonts/fontawesome-webfont.ttf
- docs/build/html/_static/css/fonts/fontawesome-webfont.woff
- docs/build/html/_static/css/fonts/fontawesome-webfont.woff2
- docs/build/html/_static/css/fonts/lato-bold-italic.woff
- docs/build/html/_static/css/fonts/lato-bold-italic.woff2
- docs/build/html/_static/css/fonts/lato-bold.woff
- docs/build/html/_static/css/fonts/lato-bold.woff2
- docs/build/html/_static/css/fonts/lato-normal-italic.woff
- docs/build/html/_static/css/fonts/lato-normal-italic.woff2
- docs/build/html/_static/css/fonts/lato-normal.woff
- docs/build/html/_static/css/fonts/lato-normal.woff2
- docs/build/html/_static/css/theme.css
- docs/build/html/_static/disqus.js
- docs/build/html/_static/doctools.js
- docs/build/html/_static/documentation_options.js
- docs/build/html/_static/file.png
- docs/build/html/_static/graphviz.css
- docs/build/html/_static/jquery-3.6.0.js
- docs/build/html/_static/jquery.js
- docs/build/html/_static/js/badge_only.js
- docs/build/html/_static/js/html5shiv-printshiv.min.js
- docs/build/html/_static/js/html5shiv.min.js
- docs/build/html/_static/js/theme.js
- docs/build/html/_static/language_data.js
- docs/build/html/_static/minus.png
- docs/build/html/_static/plus.png
- docs/build/html/_static/pygments.css
- docs/build/html/_static/searchtools.js
- docs/build/html/_static/sphinx_highlight.js
- docs/build/html/_static/underscore-1.13.1.js
- docs/build/html/_static/underscore.js
- docs/build/html/api/index.html
- docs/build/html/api/mapreader/annotate/index.html
- docs/build/html/api/mapreader/annotate/load_annotate/index.html
- docs/build/html/api/mapreader/annotate/utils/index.html
- docs/build/html/api/mapreader/download/azure_access/index.html
- docs/build/html/api/mapreader/download/index.html
- docs/build/html/api/mapreader/download/tileserver_access/index.html
- docs/build/html/api/mapreader/download/tileserver_helpers/index.html
- docs/build/html/api/mapreader/download/tileserver_scraper/index.html
- docs/build/html/api/mapreader/download/tileserver_stitcher/index.html
- docs/build/html/api/mapreader/index.html
- docs/build/html/api/mapreader/loader/images/index.html
- docs/build/html/api/mapreader/loader/index.html
- docs/build/html/api/mapreader/loader/loader/index.html
- docs/build/html/api/mapreader/process/index.html
- docs/build/html/api/mapreader/process/process/index.html
- docs/build/html/api/mapreader/slicers/index.html
- docs/build/html/api/mapreader/slicers/slicers/index.html
- docs/build/html/api/mapreader/train/classifier/index.html
- docs/build/html/api/mapreader/train/classifier_context/index.html
- docs/build/html/api/mapreader/train/custom_models/index.html
- docs/build/html/api/mapreader/train/datasets/index.html
- docs/build/html/api/mapreader/train/index.html
- docs/build/html/api/mapreader/utils/compute_and_save_stats/index.html
- docs/build/html/api/mapreader/utils/index.html
- docs/build/html/api/mapreader/utils/slice_parallel/index.html
- docs/build/html/api/mapreader/utils/utils/index.html
- docs/build/html/genindex.html
- docs/build/html/index.html
- docs/build/html/objects.inv
- docs/build/html/py-modindex.html
- docs/build/html/search.html
- docs/build/html/searchindex.js
- docs/make.bat
- docs/requirements.txt
- docs/source/Developers-guide/Developers-guide.rst
- docs/source/Input-guidance.rst
- docs/source/Install.rst
- docs/source/User-guide/Annotate.rst
- docs/source/User-guide/Download.rst
- docs/source/User-guide/Geospatial-examples.rst
- docs/source/User-guide/Load.rst
- docs/source/User-guide/Non-geospatial-examples.rst
- docs/source/User-guide/Post-process.rst
- docs/source/User-guide/Train.rst
- docs/source/User-guide/User-guide.rst
- docs/source/User-guide/Worked-examples.rst
- docs/source/api/index.rst
- docs/source/api/mapreader/annotate/index.rst
- docs/source/api/mapreader/annotate/load_annotate/index.rst
- docs/source/api/mapreader/annotate/utils/index.rst
- docs/source/api/mapreader/download/azure_access/index.rst
- docs/source/api/mapreader/download/index.rst
- docs/source/api/mapreader/download/tileserver_access/index.rst
- docs/source/api/mapreader/download/tileserver_helpers/index.rst
- docs/source/api/mapreader/download/tileserver_scraper/index.rst
- docs/source/api/mapreader/download/tileserver_stitcher/index.rst
- docs/source/api/mapreader/index.rst
- docs/source/api/mapreader/loader/images/index.rst
- docs/source/api/mapreader/loader/index.rst
- docs/source/api/mapreader/loader/loader/index.rst
- docs/source/api/mapreader/process/index.rst
- docs/source/api/mapreader/process/process/index.rst
- docs/source/api/mapreader/slicers/index.rst
- docs/source/api/mapreader/slicers/slicers/index.rst
- docs/source/api/mapreader/train/classifier/index.rst
- docs/source/api/mapreader/train/classifier_context/index.rst
- docs/source/api/mapreader/train/custom_models/index.rst
- docs/source/api/mapreader/train/datasets/index.rst
- docs/source/api/mapreader/train/index.rst
- docs/source/api/mapreader/utils/compute_and_save_stats/index.rst
- docs/source/api/mapreader/utils/index.rst
- docs/source/api/mapreader/utils/slice_parallel/index.rst
- docs/source/api/mapreader/utils/utils/index.rst
- docs/source/conf.py
- docs/source/figures/annotate.png
- docs/source/figures/annotate_context.png
- docs/source/figures/hist_published_dates.png
- docs/source/figures/loss.png
- docs/source/figures/plot_metadata_on_map.png
- docs/source/figures/show_image.png
- docs/source/figures/show_image_labels_10.png
- docs/source/figures/show_par.png
- docs/source/figures/show_par_RGB.png
- docs/source/figures/show_par_RGB_0.5.png
- docs/source/figures/show_sample_child.png
- docs/source/figures/show_sample_parent.png
- docs/source/figures/show_sample_train_8.png
- docs/source/figures/show_sample_val_8.png

#### Modified

- README.md

## [v0.3.3](https://github.com/Living-with-machines/MapReader/releases/tag/v0.3.3) (2022-04-27)

### Summary

Key actions include:

- Frequent updates to the README file to enhance clarity and organization, including restructuring content to emphasize the ongoing nature of the library and improving narrative flow.
- Changes to the project structure, such as moving map-related libraries and notebooks to designated directories and renaming categories from "maps" to "geospatial" and "non-maps" to "non-geospatial."
- Addition of new examples, specifically related to MNIST classification, including notebooks and figures.
- Incremental version updates: changing the version to 0.2.0, 0.3.0, and subsequent bug fixes culminating in versions 0.3.1 and 0.3.2.
- Improvements to Continuous Integration (CI) configuration and updates to various code files, ensuring compatibility and functionality.
- A number of bug fixes, such as addressing issues with the annotation interface for images.

Overall, the updates reflect ongoing development aimed at enhancing usability, organization, and clarity of the library and its documentation.

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

### Pull requests

- [dcsw2](https://github.com/Living-with-machines/MapReader/pull/40)
    > I've proposed removing 'originally' since we want to emphasise the ongoing nature of the library
- [dcsw2](https://github.com/Living-with-machines/MapReader/pull/41)
    > I've restored the order to 'maps' -> 'images' so we get a clearer narative as in the current existing repo; and shortened / combined a sentence, as it was repeating 'non-maps' and 'maps', so I used 'any images' instead to make it more intuitive to read.
    > 
    > I was also going to add a few sentences giving the nice positive spin about interdisciplinary cross-pollination of image analysis, but not sure where this should go: I don't want to break the flow to the instructions, so perhaps it can go after the bullet points?
- [kasra-hosseini](https://github.com/Living-with-machines/MapReader/pull/38)
    > - [x] TODOs: See https://github.com/Living-with-machines/MapReader/pull/38#issuecomment-1109569025
    > - [x] Rename Maps / Non-maps to Geospatial / Non-geospatial.
    > - [x] @kasra-hosseini Review the changes, check the links and merge.

### Files

#### Added

- assets/css/style.scss
- examples/geospatial/README.md
- examples/non-geospatial/README.md
- examples/non-geospatial/classification_mnist/001_patchify_plot.ipynb
- examples/non-geospatial/classification_mnist/002_annotation.ipynb
- examples/non-geospatial/classification_mnist/003_train_classifier.ipynb
- examples/non-geospatial/classification_mnist/annotation_tasks_mnist.yaml
- examples/non-geospatial/classification_mnist/annotations_mnist/mnist_#kasra#.csv
- figs/tutorial_classification_mnist.png
- tests/test_non_geo_pipeline.py

#### Modified

- .github/workflows/mr_ci.yml
- LICENSE
- README.md
- _config.yml
- examples/quick_start/quick_start.ipynb
- mapreader/annotate/load_annotate.py
- mapreader/annotate/utils.py
- mapreader/download/azure_access.py
- mapreader/download/tileserver_access.py
- mapreader/download/tileserver_helpers.py
- mapreader/download/tileserver_scraper.py
- mapreader/download/tileserver_stitcher.py
- mapreader/loader/images.py
- mapreader/loader/loader.py
- mapreader/process/process.py
- mapreader/slicers/slicers.py
- mapreader/train/classifier.py
- mapreader/train/classifier_context.py
- mapreader/train/datasets.py
- mapreader/utils/compute_and_save_stats.py
- mapreader/utils/slice_parallel.py
- mapreader/utils/utils.py
- setup.py
- tests/test_import.py

## [v0.1.2](https://github.com/Living-with-machines/MapReader/releases/tag/v0.1.2) (2022-03-03)

### Summary

#### Commit Highlights:
- **Instructions Added**: Enhanced kernel installation instructions included.
- **Merging**: Several branches were merged to consolidate changes across the project.
- **Requirements Updated**: The `requirements.txt` and related files were updated multiple times to ensure dependencies are current.
- **Package Management**: Added and subsequently removed a package for BinderHub and updated `pyproj` versions.
- **Enhancements**: Introduced new features such as `max_mean_pixel` in the annotation interface and included a quick start notebook with practical examples.
- **File Structure Changes**: A `setup.py` file was added, while `poetry.lock` and `pyproject.toml` were removed, indicating a shift in dependency management strategy.

#### Pull Requests:
- Notable contributions from the user kasra-hosseini that address specific issues related to model inference.

#### File Changes:
- **Added**: Several files for example datasets and a quick start guide.
- **Modified**: Key files including CI workflows, the README, and utility scripts were updated.
- **Removed**: The `poetry.lock` and `pyproject.toml` files were removed.

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

### Pull requests

- [kasra-hosseini](https://github.com/Living-with-machines/MapReader/pull/24)

- [kasra-hosseini](https://github.com/Living-with-machines/MapReader/pull/25)
    > Issue https://github.com/Living-with-machines/MapReader/issues/19

### Files

#### Added

- examples/quick_start/annotation_tasks_open.yaml
- examples/quick_start/annotations_phenotype_open_access/phenotype_test_#kasra#.csv
- examples/quick_start/dataset/open_access_plant/2014-06-06_plant001_rgb.png
- examples/quick_start/dataset/open_access_plant/2014-07-17_plant047_rgb.png
- examples/quick_start/quick_start.ipynb
- setup.py

#### Modified

- .github/workflows/mr_ci.yml
- README.md
- mapreader/annotate/utils.py
- mapreader/train/classifier_context.py

#### Removed

- poetry.lock
- pyproject.toml

## [v0.1.1](https://github.com/Living-with-machines/MapReader/releases/tag/v0.1.1) (2022-01-15)

The first published version of MapReader.
