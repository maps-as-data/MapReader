# MapReader and geospatial images 
## Examples of using MapReader with scanned maps or earth observation imagery

### Worked Examples 

These subdirectories contain jupyter notebooks that use MapReader for different tasks.

| Worked Example Name  | Task Type | Input Type | Link to Input Data Source | Brief Description of What Notebook Does | Output Type | Model | Related Paper | Created By |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| classification_one_inch_maps  | Patch Classification| Historical Maps | https://cloud.maptiler.com/tiles/uk-osgb63k1885/ | Classify patches from OS 1-inch maps | MapReader patch output | None (created in notebook) | (no paper)| Kasra Hosseini, Rosie Wood |
| context_classification_one_inch_maps | Patch Classification: Context Model | Historical Maps | https://cloud.maptiler.com/tiles/uk-osgb63k1885/ | Classify patches from OS 1-inch maps using context models | MapReader patch outputs | None (created in notebook) | (no paper) | Rosie Wood |
|text_spotting_one_inch_maps | Text Spotting | Historical Maps | https://cloud.maptiler.com/tiles/uk-osgb63k1885/ | Detect & Recognize text on OS 1-inch maps | MapReader patch outputs | https://github.com/rwood-97/DeepSolo/tree/dev and https://github.com/rwood-97/DPText-DETR | (no paper) | Rosie Wood |
| workshop_april_2024 | Patch Classification | Historical Maps | https://cloud.maptiler.com/tiles/uk-osgb10k1888/ | Classify patches from OS 6-inch 2nd edition maps | MapReader patch outputs | None (created in notebook), also uses https://huggingface.co/Livingwithmachines/mr_resnest101e_finetuned_OS_6inch_2nd_ed_railspace | (no paper) | Rosie Wood |
| workshop_june_2023 | Patch Classification | Historical Maps | https://cloud.maptiler.com/tiles/uk-osgb63k1885/  | Classify patches from OS maps | MapReader patch outputs | None (created in notebook) | (no paper) | Rosie Wood, Katie McDonough |
| annotation_examples | Patch Classification | Historical Maps | https://cloud.maptiler.com/tiles/uk-osgb1888/ | Annotate patches from OS maps | MapReader annotations in csv format | None | (no paper) | Rosie Wood, Katie McDonough |
<!--| Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  |-->


### Worked Example Required Files

These files are located within the worked_examples directory because they are required to run various notebooks provided here.

| Required Files Name  |  File Type | Link to File Example | Brief Description of File | Created By |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| metadata_OS_Six_Inch_GB_WFS_light.json | json | [6-inch OS metadata](https://github.com/Living-with-machines/MapReader/blob/main/worked_examples/geospatial/NLS_metadata/metadata_OS_Six_Inch_GB_WFS_light.json) | Metadata for different historical map collections used in worked examples | National Library of Scotland  | 
| metadata_OS_One_Inch_GB_WFS_light.json  | json | [1-inch OS metadata](https://github.com/Living-with-machines/MapReader/blob/main/worked_examples/geospatial/NLS_metadata/metadata_OS_One_Inch_GB_WFS_light.json) | Metadata for different historical map collections used in worked examples | National Library of Scotland  | 
<!--| Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  |-->

#### How can I contribute a worked example?
Open a ["Documentation Update"](https://github.com/Living-with-machines/MapReader/issues/new?assignees=&labels=documentation&projects=&template=documentation_update.md&title=) ticket and pitch your contribution to the team!

