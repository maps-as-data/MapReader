# MapReader and maps

## Tutorials

- [classification_one_inch_maps_001](https://github.com/Living-with-machines/MapReader/tree/main/examples/maps/classification_one_inch_maps_001)
  * **Goal:** train/fine-tune PyTorch CV classifiers on historical maps.
  * **Dataset:** from National Library of Scotland: [OS one-inch, 2nd edition layer](https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/index.html).
  * **Data access:** tileserver
  * **Annotations** are done on map patches (i.e., slices of each map).
  * **Classifier:** train/fine-tuned PyTorch CV models.


## Guidance for specific user groups

##### "I want to search the visual contents of a large set of maps to help me answer a question about the past."

- MapReader can help you find instances of spatial phenomena in a collection of maps that is too large for you to 'close read/view'.
- MapReader creates output that you can link and analyze in relation to other geospatial datasets (e.g. census, gazetteers, toponyms in text corpora).

##### "I have a collection of maps that have been scanned and georeferenced. How can I use MapReader?"

- If your maps cannot be openly released, MapReader can be used to create derived data that can be shared publicly. The institution could create these datasets or individual researchers could create datasets specific to their research questions.
