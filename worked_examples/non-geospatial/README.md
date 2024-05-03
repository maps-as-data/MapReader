# MapReader and non-geospatial images

## Worked Examples

- [classification_plant_phenotype](./classification_plant_phenotype)
  * **Goal:** train/fine-tune PyTorch CV classifiers on plant patches in images (plant phenotyping example).
  * **Dataset:** Example images taken from the openly accessible `CVPPP2014_LSV_training_data` dataset available from https://www.plant-phenotyping.org/datasets-download.
  * **Data access:** locally stored
  * **Annotations** are done on plant patches (i.e., slices of each plant image).
  * **Classifier:** train/fine-tuned PyTorch CV models.
 
  * 
- [classification_mnist](./classification_mnist)
  * **Goal:** train/fine-tune PyTorch CV classifiers on MNIST.
  * **Dataset:** Example images taken from http://yann.lecun.com/exdb/mnist/.
  * **Data access:** locally stored
  * **Annotations** are done on whole MNIST images, **not** on patches/slices of those images.
  * **Classifier:** train/fine-tuned PyTorch CV models.

| Worked Example Name  | Task Type | Input Type | Link to Input Data | Brief Description of What Notebook Does | Output Type | Created By | Annotations | Model | Paper |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | 
| classification_plant_phenotype | Patch Classification | Plant Images | TBC | Fine-tune PyTorch CV classifiers on plant patches in images  | MapReader patch classifications in csv format | 
 Evangeline Corcoran, Kasra Hosseini | TBC  | TBC  | TBC | 
| classification_mnist  | Image Classification | MNIST  | http://yann.lecun.com/exdb/mnist/ | train/fine-tune PyTorch CV classifiers on MNIST | MapReader whole image classifications in csv format  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | 
