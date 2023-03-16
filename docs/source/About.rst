About MapReader
================

What is MapReader?
-------------------

MapReader is an end-to-end computer vision (CV) pipeline for exploring and analyzing images at scale.

The 'pipeline'
~~~~~~~~~~~~~~~

The MapReader pipeline consists of a linear sequence of tasks which can be used to:

1. **Download** images/maps and metadata stored on web-servers (e.g. tileserves which can be used to retrieve maps from OpenStreetMap (OSM), the National Library of Scotland (NLS), or elsewhere).
2. **Load** images/maps and metadata stored locally and **pre-process** these:
   - 'patchify' images/maps (i.e. slice a whole image/map into smaller patches for analysis),
   - resample images/maps (use image transformations to alter pixel-dimensions/resolution/orientation/etc.),
   - remove borders outside the neatline,
   - reproject maps between coordinate reference systems (CRS).
3. **Annotate** (i.e. add labels to) images/maps or their patches using an interactive annotation tool.
4. **Train or fine-tune** a computer vision (CV) classifier to recognise image features associated with these labels.
5. Use inference to **predict labels** on large sets of images/maps or their patches. 
6. **Post-process**, analyse and visualise outputs.

.. image:: figures/pipeline_explained.png

Why use MapReader?
-------------------

**Do you want to search the visual contents of a large set of maps to answer a question about the past?**
MapReader can help you find instances of spatial phenomena in a collection of maps that is too large for you to 'close read/view'.

MapReader enables quick, flexible research with large map corpora. 
It is based on the patchwork method, e.g. scanned map sheets are preprocessed to divide up the content of the map into a grid of squares, or "patches". 

Using image classificaion at the level of each patch allows users to define classes (labels) of features on maps related to their research questions.

MapReader creates output that you can link and analyze in relation to other geospatial datasets (e.g. census, gazetteers, toponyms in text corpora).

You might be interested in using MapReader if...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* you have access to a large corpus of georeferenced maps (MapReader can still provide results for non-georeferenced maps)
* you want to quickly test different labels to help refine a research question that depends on finding content on maps before/without committing to manual vector data creation
* your maps were created before surveying accuracy reached modern standards, and therefore you do not want to create overly precise geolocated data based on the content of those maps 

MapReader is well-suited for finding spatial phenomena that...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* have a homogeneous visual signal across many maps in the same style
* may not correspond to typical categories of map features that are traditionally digitized as vector data in a GIS
* you then want to combine with other geospatial datasets 

Skills/knowledge you will need to use MapReader
-------------------------------------------------

* Basic understanding of how to use your terminal
* Basic python
* Basic understanding of machine learning and computer vision methodology