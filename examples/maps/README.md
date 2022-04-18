# MapReader and maps

- [Tutorials](#tutorials)
  - [classification_one_inch_maps_001](https://github.com/Living-with-machines/MapReader/tree/main/examples/maps/classification_one_inch_maps_001)
- [Why use MapReader?](#why-use-mapreader)
- [Guidance for specific user groups](#guidance-for-specific-user-groups)

## Tutorials

- [classification_one_inch_maps_001](https://github.com/Living-with-machines/MapReader/tree/main/examples/maps/classification_one_inch_maps_001)
  * **Goal:** train/fine-tune PyTorch CV classifiers on historical maps.
  * **Dataset:** from National Library of Scotland: [OS one-inch, 2nd edition layer](https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/index.html).
  * **Data access:** tileserver
  * **Annotations** are done on map patches (i.e., slices of each map).
  * **Classifier:** train/fine-tuned PyTorch CV models.

## Why use MapReader?

MapReader enables quick, flexible research with large map corpora. It is based on the patchwork method, e.g. scanned map sheets are preprocessed to divide up the content of the map into a grid of squares. Using image classificaion at the level of each patch allows users to define classes (labels) of features on maps related to their research questions. 

#### You might be interested in using MapReader if:
- you have access to a large corpora of georeferenced maps
- you want to quickly test different labels to help refine your research question before/without committing to manual vector data creation
- your maps were created before surveying accuracy reached modern standards, and therefore you do not want to create overly precise geolocated data based on the content of those maps

#### MapReader is well-suited for finding spatial phenomena that:
- have a homogeneous visual signal across many maps 
- may not correspond to typical categories of map features that are traditionally digitized as vector data in a GIS


## Guidance for specific user groups

##### "I want to search the visual contents of a large set of maps to help me answer a question about the past."

- MapReader can help you find instances of spatial phenomena in a collection of maps that is too large for you to 'close read/view'.
- MapReader creates output that you can link and analyze in relation to other geospatial datasets (e.g. census, gazetteers, toponyms in text corpora).

##### "I have a collection of maps that have been scanned and georeferenced. How can I use MapReader?"

- If your maps cannot be openly released, MapReader can be used to create derived data that can be shared publicly. The institution could create these datasets or individual researchers could create datasets specific to their research questions.
