:py:mod:`mapreader.download.azure_access`
=========================================

.. py:module:: mapreader.download.azure_access


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   mapreader.download.azure_access.initBlobServiceClient
   mapreader.download.azure_access.initContainerClient
   mapreader.download.azure_access.downloadClient
   mapreader.download.azure_access.dirnamesBlobs



.. py:function:: initBlobServiceClient(env_variable = 'AZURE_STORAGE_CONNECTION_STRING')

   Initialize a BlobServiceClient object from the Azure Storage connection
   string.

   Parameters
   ----------
   env_variable : str, optional
       The name of the environment variable that contains the Azure Storage
       connection string. Default is ``"AZURE_STORAGE_CONNECTION_STRING"``.

   Returns
   -------
   blob_service_client : azure.storage.blob.BlobServiceClient
       An instance of the ``BlobServiceClient`` class.

   Raises
   ------
   ValueError
       If the environment variable ``AZURE_STORAGE_CONNECTION_STRING`` is not
       found or empty.

   Notes
   ------
   The environment variable ``AZURE_STORAGE_CONNECTION_STRING`` needs to be
   set before running this function.


.. py:function:: initContainerClient(blob_client, container)

   Initialize a ``ContainerClient`` object for a specified container and list
   all blobs in that container.

   Parameters
   ----------
   blob_client : azure.storage.blob.BlobClient
       An instance of the ``BlobClient`` class.
   container : str
       The name of the container.

   Returns
   -------
   container_client : azure.storage.blob.ContainerClient
       An instance of the ``ContainerClient`` class for the specified
       container.
   list_containers : list
       A list of ``azure.storage.blob.BlobProperties`` objects representing
       the blobs in the container.


.. py:function:: downloadClient(container_client, list_containers, download_dir, include_dirnames = None, nls_names = None, max_download = None, index1 = 0, index2 = None, force_download = False)

   Download blobs from a container and save them in a local directory.

   Parameters
   ----------
   container_client : azure.storage.blob.ContainerClient
       An instance of the ``ContainerClient`` class for the specified
       container, initialised by ``initContainerClient`` function.
   list_containers : list
       A list of ``azure.storage.blob.BlobProperties`` objects representing
       the blobs in the container, also initialised by ``initContainerClient``
       function.
   download_dir : str
       The directory to which downloaded files will be saved.
   include_dirnames : str or list of str, optional
       A list of directory names to include in the download. Default is
       ``None`` (all directories are included). The dirnames can be printed
       by the :func:`mapreader.download.azure_access.dirnamesBlobs` function.
   nls_names : str or list of str, optional
       A list of file names (without extensions) from the
       https://maps.nls.uk/geo/ website to include in the download. Default
       is ``None`` (all files are included).
   max_download : int, optional
       The maximum number of files to download. Default is ``None`` (download
       all files).
   index1 : int, optional
       The index of the first file to download. Default is ``0``.
   index2 : int, optional
       The index of the last file to download. Default is ``None`` (download
       until the end of the list).
   force_download : bool, optional
       If ``True``, force download even if the file already exists in the
       download directory. Default is ``False``.

   Returns
   -------
   None


.. py:function:: dirnamesBlobs(list_containers)

   Given a list of blob containers, return a list of unique directory names
   containing the blobs.

   Parameters
   ----------
   list_containers : list
       A list of blob containers.

   Returns
   -------
   all_dirs : list
       A list of unique directory names containing the blobs.

   ..
       TODO: The documentation here needs to have a different type for the
       List[Any] above. What would it be?


