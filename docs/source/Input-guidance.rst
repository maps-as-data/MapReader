===============
Input guidance
===============


.. contents:: Table of contents


******************
Why use MapReader?
******************

**Do you want to search the visual contents of a large set of maps to answer a question about the past?**
MapReader can help you find instances of spatial phenomena in a collection of maps that is too large for you to ‘close read/view’.

MapReader enables quick, flexible research with large map corpora. It is based on the patchwork method, e.g. scanned map sheets are preprocessed to divide up the content of the map into a grid of squares, or "patches". 

Using image classificaion at the level of each patch allows users to define classes (labels) of features on maps related to their research questions.

MapReader creates output that you can link and analyze in relation to other geospatial datasets (e.g. census, gazetteers, toponyms in text corpora).

You might be interested in using MapReader if...
---------------------------------------------
* you have access to a large corpus of georeferenced maps (For non-georeferenced maps, MapReader can still provide results. Please see below.)
* you want to quickly test different labels to help refine a research question that depends on finding content on maps before/without committing to manual vector data creation
* your maps were created before surveying accuracy reached modern standards, and therefore you do not want to create overly precise geolocated data based on the content of those maps 

MapReader is well-suited for finding spatial phenomena that...
--------------------------------------------------------------
* have a homogeneous visual signal across many maps in the same style
* may not correspond to typical categories of map features that are traditionally digitized as vector data in a GIS
* you then want to combime with other geospatial datasets 


*************************
Preparing your map corpus
*************************

How many maps?
=============

MapReader was designed to help researchers work with large collections of series maps. Deciding to use MapReader, which uses deep learning computer vision models to predict the class of content on patches across many sheets, means weighing the pros and cons of working with the data output that is inferred by the model. Inferred data can be evaluated against expert-annotated data to understand its general quality (are all instances of a feature of interest identified by the model? does the model apply the correct label to that feature?), but in the full dataset there *will necessarily be* some percentage of error. 

So, MapReader is useful when the number of maps you wish to analyze exceeds the number which you (or a team) might be willing and able to annotate manually, using tools like ArcGIS, QGIS, or web-based annotation interfaces like Recogito. This number will vary depending on the size of your maps, the features you want to find, the skills you and your team have, and the amount of time at your disposal. 

Image file format, quality, and size
====================================

MapReader accepts different formats of digitized maps as input:
a. jpeg or png files with associated files containing georeferencing metadata for each sheet
b. geoTIFFs containing georeferencing metadata
c. layers from tileservers
d. non-georeferenced map scans (e.g. jpegs, png, tiff with no metadata to locate the sheet on the earth)

Image file directory structure
==============================

*Work in progress*


***********************
Preparing your metadata
***********************

*Work in progress*
