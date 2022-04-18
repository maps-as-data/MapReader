# MapReader and non-map images

Tutorials:

- [classification_plant_phenotype](https://github.com/Living-with-machines/MapReader/tree/main/examples/non-maps/classification_plant_phenotype)
  * **Goal:** train/fine-tune PyTorch CV classifiers on plant patches in images (plant phenotyping example).
  * **Dataset:** Example images taken from the openly accessible `CVPPP2014_LSV_training_data` dataset available from https://www.plant-phenotyping.org/datasets-download. 
  * **Data access:** locally stored
  * **Annotations** are done on plant patches (i.e., slices of each plant image).
  * **Classifier:** train/fine-tuned PyTorch CV models.
- [classification_mnist](https://github.com/Living-with-machines/MapReader/tree/main/examples/non-maps/classification_mnist)
  * **Goal:** train/fine-tune PyTorch CV classifiers on MNIST.
  * **Dataset:** Example images taken from http://yann.lecun.com/exdb/mnist/. 
  * **Data access:** locally stored
  * **Annotations** are done on MNIST (NOT patches/slices of MNIST images).
  * **Classifier:** train/fine-tuned PyTorch CV models.
