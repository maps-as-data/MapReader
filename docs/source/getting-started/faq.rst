Frequently Asked Questions
==========================

Welcome to the Frequently Asked Questions (FAQ) section of our documentation!
Here, we address some of the most common queries and challenges faced by users of MapReader that we've encountered when we've led workshops or during our community calls.
Whether you're just starting with a MapReader project or diving into our more advanced features or contributing code, this section provides quick answers and solutions to help you navigate effectively.

If you have a question that is not covered here, feel free to reach out to open an issue on our GitHub repository or reach out to us elsewhere.
(If you don't know how to use GitHub, we have `a section for that in the documentation </in-depth-resources/github-basics.html>`_ as well!)

We're always here to help and welcome contributions to expand this section.

..
    Once we have too many questions, we can start integrating them into categories (e.g., "General Questions," "Data Preparation," "Model Training," etc.)

        General Questions
        -----------------

**Q: Do I need georeferenced maps?**

A: MapReader is at its most useful when operating on georeferenced maps, as these allow for geospatial analysis and visualization.
However, it is possible to use non-georeferenced maps as input.
Output from non-georeferenced maps will provide details about patches or text bounding boxes based only on pixel coordinates, not geospatial coordinates.
In practice, this means that you can analyze information about patches within 1 map, or across maps based on quantiative measures (for example, how many of each label? where in the image do these labeled patches occur? what text occurs most frequently across my corpus).
But you are not able to plot this output on a map and perform spatial analysis.

Note that you can work with non-georeferenced maps in MapReader, georeference the maps separately (for example, in QGIS or with Allmaps), and then transform the MapReader output coordinates from pixels into geospatial locations after the fact.

**Q: Do you have good examples of maps that I can start working with?**

A: Yes.
MapReader takes different kinds of input formats for maps.
Details are available in the `File/Map Options </using-mapreader/input-guidance/file-map-options.html>`_ section.
Map collections that are available as tiles, IIIF resources, and direct image downloads include:

- **Tiles**

  - National Library of Scotland maps as web map tiles: https://www.maptiler.com/nls/

- **IIIF Map Collections**

  - National Library of Scotland maps as IIIF resources (when viewing 1 map, click on "Show IIIF" tab at bottom of frame)
  - David Rumsey Map Collection: https://www.davidrumsey.com/ (via the "Share" menu)
  - Library of Congress: https://www.loc.gov/maps/?fa=partof:geography+and+map+division (scroll down to "IIIF Presentation Manifest")
  - John Carter Brown Library Map Collection: https://jcb.lunaimaging.com/luna/servlet/JCBMAPS~1~1 (same infrastructure as Rumsey collection for finding IIIF manifests)
  - `Bibliotheque nationale de France/Gallica digital map collection <https://gallica.bnf.fr/services/engine/search/sru?operation=searchRetrieve&exactSearch=false&collapsing=true&version=1.2&query=(dc.type%20all%20%22carte%22)&suggest=10&keywords=>`_

- **Direct Downloads**

  - David Rumsey Map Collection: https://www.davidrumsey.com/ (scroll down, to 'download' options in metadata field, or use 'export' option in top right)
  - Library of Congress: https://www.loc.gov/maps/?fa=partof:geography+and+map+division
  - John Carter Brown Library Map Collection: https://jcb.lunaimaging.com/luna/servlet/JCBMAPS~1~1 (export option in top right)
  - `Bibliotheque nationale de France/Gallica digital map collection <https://gallica.bnf.fr/services/engine/search/sru?operation=searchRetrieve&exactSearch=false&collapsing=true&version=1.2&query=(dc.type%20all%20%22carte%22)&suggest=10&keywords=>`_

**Q: Can the tool download/patchify IIIF maps?**

A: MapReader does not yet support downloading and patchifying IIIF maps.
We are working on an integration of IIIF-based map servers with the tool to download and patchify maps as needed.
It is expected to be delivered early 2025.

**Q: Does MapReader have a GUI?**

A: No, MapReader does not currently have a graphical user interface.
It is designed to be used via Python scripts or Jupyter notebooks, offering flexibility for advanced users.

**Q: How much technical background do we need to have?**

A: Basic familiarity with Python and Jupyter notebooks is recommended.
Knowledge of geospatial data handling is a plus but not mandatory for beginners.
We have provided some `coding basics </in-depth-resources/coding-basics/>`_ in this documentation site to help you get started.

**Q: What does the output of MapReader look like?**

A: MapReader produces patch data, annotations, text spotting outputs, as well as model predictions.
Outputs can include CSV files for metadata and labels, annotated patches in image formats, and trained models for further inference.

**Q: How can I contribute to the codebase?**

A: Contributions are welcome!
You can fork the MapReader repository, make changes, and submit a pull request.
Please review our `code contribution guide </community-and-contributions/contribution-guide/getting-started-with-contributions/add-or-update-code>`_ for details.

**Q: Can I use MapReader to vectorize maps?**

A: MapReader is not designed for full vectorization but can assist in creating training datasets for machine learning models, which can then infer vector-like features.

**Q: Is it free to use MapReader?**

A: Yes, MapReader is an open-source project and is free to use under its license.

**Q: Why do I need to patchify my maps?**

A: Patchifying maps is a crucial step in MapReader because it breaks down large map sheets into smaller, manageable regions called patches.
Without the step of patchifying, processing large, complex maps would be resource-intensive and less flexible for targeted analysis or model training.
Overall, this step is important as it enables some crucial features for you:

- *Efficient data analysis*:
  Analyzing smaller patches reduces computational overhead, making tasks like classification, annotation, and visualization more scalable.
- *Better machine learning models*:
  Most machine learning models process data in fixed-size inputs.
  Patchifying maps ensures that the data conforms to the required dimensions for training and inference.
- *Adaptability to the patch's context*:
  Patchifying enables the use of context-based models, where information from surrounding patches can be incorporated for better predictions.

**Q: How large do my patches need to be? (i.e., standards? best patch sizes?)**

A: Patch sizes depend on the resolution of your maps and your analysis goals.
Standard sizes are 100x100 or 300x300 pixels, but experimentation may be necessary for optimal results.
You can read more in the `"Patchify" section </using-mapreader/step-by-step-guide/2-load.html#patchify>`_ in our step-by-step guide.

**Q: What should I use for labels? What is a good label?**

A: Labels should correspond to a visual signal you know is present in the maps you are working with.
This visual signal can have some variety, but too much variety will lead to poor results.
Developing the name of a label and the guidelines for assigning a label to a patch is usually an iterative process that is refined as you start to annotate patches.
Labels do not need to conform to typical map features, but they can.
Previous examples of patch labels include "railspace", "building", "trees", "rocks", "water", "gardens", and "heath".

**Q: How can I trust the output from MapReader?**

A: Trust in MapReader's output relies on the quality of your training data and the rigor of your validation process.
Review results critically and iteratively refine models and annotations for accuracy.
We have some post-processing tools to help with this, such as the :class:`~.process.occlusion_analysis.OcclusionAnalyzer` class.

**Q: Do I need Cartopy? What does it do in MapReader? What happens if I don't have it?**

A: No, Cartopy is optional. It is used for plotting geospatial data on maps (e.g., via matplotlib) in methods like :meth:`~.download.sheet_downloader.SheetDownloader.plot_all_metadata_on_map` within the Download subpackage.

   - **With Cartopy**: Geospatial plots can display enhanced geographic context (e.g., basemaps).
   - **Without Cartopy**: MapReader will still function, but the plots will lack certain geospatial overlays.
