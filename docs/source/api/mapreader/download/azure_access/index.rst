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



Attributes
~~~~~~~~~~

.. autoapisummary::

   mapreader.download.azure_access.LINK_SETUP_AZURE


.. py:data:: LINK_SETUP_AZURE
   :value: 'https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python#configure-yo...'

   

.. py:function:: initBlobServiceClient(env_variable='AZURE_STORAGE_CONNECTION_STRING')

   Instantiate BlobServiceClient

   Note: $AZURE_STORAGE_CONNECTION_STRING needs to be set.


.. py:function:: initContainerClient(blob_client, container)

   Instantiate a container client

   Arguments:
       blob_client -- Blob Storage Client initialized by initBlobServiceClient function
       container {str} -- Container to access/download from.


.. py:function:: downloadClient(container_client, list_containers, download_dir, include_dirnames=None, nls_names=None, max_download=None, index1=0, index2=None, force_download=False)

   Download client using a client container

   Arguments:
       container_client -- Container client initialized by initContainerClient function
       list_containers {list} -- list of all blobs in the container, also initialized by initContainerClient function
       download_dir {path} -- path to save downloaded files

   Keyword Arguments:
       include_dirnames {list, None} -- only include this list of dirnames. The dirnames can be printed by the dirnamesBlobs function.
                                        if None, all directories will be considered.
       nls_names {list, None} -- list of NLS names from https://maps.nls.uk/geo/ website
       max_download {int, None} -- Max. number of files to be downloaded
       index1 {int} -- first index to be used to download blobs with indices [index1, index2) (default: {0})
       index2 {init, None} -- similar to index1 except for the max. bound. If None, download [index1:len(list_containers)) (default: {None})
       force_download {bool} -- re-download all files as specified by index1 and index2 (default: {False})


.. py:function:: dirnamesBlobs(list_containers)

   Print all dirnames in list_containers


