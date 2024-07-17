Why should you use MapReader?
=============================

MapReader becomes useful when the number of maps you wish to analyze exceeds the number which you (or your team) are willing to/capable of annotating manually.

This exact number will vary depending on:

- the size of your maps,
- the features you want to find,
- the skills you (or your team) have,
- the amount of time at your disposal.

Deciding to use MapReader, which uses deep learning computer vision (CV) models to predict the class of content on patches across many sheets, means weighing the pros and cons of working with the data output that is inferred by the model.
Inferred data can be evaluated against expert-annotated data to understand its general quality (are all instances of a feature of interest identified by the model? does the model apply the correct label to that feature?), but in the full dataset there *will necessarily be* some percentage of error.

MapReader creates output that you can link and analyze in relation to other geospatial datasets (e.g. census, gazetteers, toponyms in text corpora).
