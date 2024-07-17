mapreader.classify.load_annotations
===================================

.. py:module:: mapreader.classify.load_annotations


Classes
-------

.. autoapisummary::

   mapreader.classify.load_annotations.AnnotationsLoader


Module Contents
---------------

.. py:class:: AnnotationsLoader

   .. py:method:: load(annotations, delimiter = ',', images_dir = None, remove_broken = True, ignore_broken = False, patch_paths_col = 'image_path', label_col = 'label', append = True, scramble_frame = False, reset_index = False)

      Loads annotations from a csv file or dataframe and can be used to set the ``patch_paths_col`` and ``label_col`` attributes.

      :param annotations: The annotations.
                          Can either be the path to a csv file or a pandas.DataFrame.
      :type annotations: Union[str, pd.DataFrame]
      :param delimiter: The delimiter to use when loading the csv file as a dataframe, by default ",".
      :type delimiter: Optional[str], optional
      :param images_dir: The path to the directory in which patches are stored.
                         This argument should be passed if image paths are different from the path saved in annotations dataframe/csv.
                         If None, no updates will be made to the image paths in the annotations dataframe/csv.
                         By default None.
      :type images_dir: Optional[str], optional
      :param remove_broken: Whether to remove annotations with broken image paths.
                            If False, annotations with broken paths will remain in annotations dataframe and may cause issues!
                            By default True.
      :type remove_broken: Optional[bool], optional
      :param ignore_broken: Whether to ignore broken image paths (only valid if remove_broken=False).
                            If True, annotations with broken paths will remain in annotations dataframe and no error will be raised. This may cause issues!
                            If False, annotations with broken paths will raise error. By default, False.
      :type ignore_broken: Optional[bool], optional
      :param patch_paths_col: The name of the column containing the image paths, by default "image_path".
      :type patch_paths_col: Optional[str], optional
      :param label_col: The name of the column containing the image labels, by default "label".
      :type label_col: Optional[str], optional
      :param append: Whether to append the annotations to a pre-existing ``annotations`` dataframe.
                     If False, existing dataframe will be overwritten.
                     By default True.
      :type append: Optional[bool], optional
      :param scramble_frame: Whether to shuffle the rows of the dataframe, by default False.
      :type scramble_frame: Optional[bool], optional
      :param reset_index: Whether to reset the index of the dataframe (e.g. after shuffling), by default False.
      :type reset_index: Optional[bool], optional

      :raises ValueError: If ``annotations`` is passed as something other than a string or pd.DataFrame.



   .. py:method:: show_patch(patch_id)

      Display a patch and its label.

      :param patch_id: The image ID of the patch to show.
      :type patch_id: str

      :rtype: None



   .. py:method:: print_unique_labels()

      Prints unique labels

      :raises ValueError: If no annotations are found.



   .. py:method:: review_labels(label_to_review = None, chunks = 8 * 3, num_cols = 8, exclude_df = None, include_df = None, deduplicate_col = 'image_id')

      Perform image review on annotations and update labels for a given
      label or all labels.

      :param label_to_review: The target label to review. If not provided, all labels will be
                              reviewed, by default ``None``.
      :type label_to_review: str, optional
      :param chunks: The number of images to display at a time, by default ``24``.
      :type chunks: int, optional
      :param num_cols: The number of columns in the display, by default ``8``.
      :type num_cols: int, optional
      :param exclude_df: A DataFrame of images to exclude from review, by default ``None``.
      :type exclude_df: pandas.DataFrame, optional
      :param include_df: A DataFrame of images to include for review, by default ``None``.
      :type include_df: pandas.DataFrame, optional
      :param deduplicate_col: The column to use for deduplicating reviewed images, by default
                              ``"image_id"``.
      :type deduplicate_col: str, optional

      :rtype: None

      .. rubric:: Notes

      This method reviews images with their corresponding labels and allows
      the user to change the label for each image.

      Updated labels are saved in ``self.annotations`` and in a newly created ``self.reviewed`` DataFrame.
      If ``exclude_df`` is provided, images found in this df are skipped in the review process.
      If ``include_df`` is provided, only images found in this df are reviewed.
      The ``self.reviewed`` DataFrame is deduplicated based on the ``deduplicate_col``.



   .. py:method:: show_sample(label_to_show, num_samples = 9)

      Show a random sample of images with the specified label (tar_label).

      :param label_to_show: The label of the images to show.
      :type label_to_show: str, optional
      :param num_sample: The number of images to show.
                         If ``None``, all images with the specified label will be shown. Default is ``9``.
      :type num_sample: int, optional

      :rtype: None



   .. py:method:: create_datasets(frac_train = 0.7, frac_val = 0.15, frac_test = 0.15, random_state = 1364, train_transform = 'train', val_transform = 'val', test_transform = 'test', context_datasets = False, context_df = None)

      Splits the dataset into three subsets: training, validation, and test sets (DataFrames) and saves them as a dictionary in ``self.datasets``.

      :param frac_train: Fraction of the dataset to be used for training.
                         By default ``0.70``.
      :type frac_train: float, optional
      :param frac_val: Fraction of the dataset to be used for validation.
                       By default ``0.15``.
      :type frac_val: float, optional
      :param frac_test: Fraction of the dataset to be used for testing.
                        By default ``0.15``.
      :type frac_test: float, optional
      :param random_state: Random seed to ensure reproducibility. The default is ``1364``.
      :type random_state: int, optional
      :param train_transform: The transform to use on the training dataset images.
                              Options are "train", "test" or "val" or, a callable object (e.g. a torchvision transform or torchvision.transforms.Compose).
                              By default "train".
      :type train_transform: str, tochvision.transforms.Compose or Callable, optional
      :param val_transform: The transform to use on the validation dataset images.
                            Options are "train", "test" or "val" or, a callable object (e.g. a torchvision transform or torchvision.transforms.Compose).
                            By default "val".
      :type val_transform: str, tochvision.transforms.Compose or Callable, optional
      :param test_transform: The transform to use on the test dataset images.
                             Options are "train", "test" or "val" or, a callable object (e.g. a torchvision transform or torchvision.transforms.Compose).
                             By default "test".
      :type test_transform: str, tochvision.transforms.Compose or Callable, optional
      :param context_datasets: Whether to create context datasets or not. By default False.
      :type context_datasets: bool, optional
      :param context_df: The dataframe containing all patches if using context datasets.
                         Used to create context images. By default None.
      :type context_df: str or pandas.DataFrame, optional

      :raises ValueError: If the sum of fractions of training, validation and test sets does
          not add up to 1.

      :rtype: None

      .. rubric:: Notes

      This method saves the split datasets as a dictionary in ``self.datasets``.

      Following fractional ratios provided by the user, where each subset is
      stratified by the values in a specific column (that is, each subset has
      the same relative frequency of the values in the column). It performs
      this splitting by running ``train_test_split()`` twice.

      See ``PatchDataset`` for more information on transforms.



   .. py:method:: create_patch_datasets(train_transform, val_transform, test_transform, df_train, df_val, df_test)


   .. py:method:: create_patch_context_datasets(context_df, train_transform, val_transform, test_transform, df_train, df_val, df_test)


   .. py:method:: create_dataloaders(batch_size = 16, sampler = 'default', shuffle = False, num_workers = 0, **kwargs)

      Creates a dictionary containing PyTorch dataloaders
      saves it to as ``self.dataloaders`` and returns it.

      :param batch_size: The batch size to use for the dataloader. By default ``16``.
      :type batch_size: int, optional
      :param sampler: The sampler to use when creating batches from the training dataset.
      :type sampler: Sampler, str or None, optional
      :param shuffle: Whether to shuffle the dataset during training. By default ``False``.
      :type shuffle: bool, optional
      :param num_workers: The number of worker threads to use for loading data. By default ``0``.
      :type num_workers: int, optional
      :param \*\*kwds: Additional keyword arguments to pass to PyTorch's ``DataLoader`` constructor.

      :returns: Dictionary containing dataloaders.
      :rtype: Dict

      .. rubric:: Notes

      ``sampler`` will only be applied to the training dataset (datasets["train"]).
