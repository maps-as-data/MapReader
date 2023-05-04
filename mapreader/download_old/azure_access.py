#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

try:
    from azure.storage.blob import (
        BlobServiceClient,
        BlobClient,
        ContainerClient,
    )
except ImportError:
    raise ImportError(
        "Please install 'azure-storage-blob'.\n"
        "Link: https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python#install-the-package"  # noqa
    )

from typing import Optional, Tuple, List, Union, Any

# Keep this outside of the function so it is easier to change/maintain
_LINK_SETUP_AZURE = "https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python#configure-your-storage-connection-string"  # noqa


def initBlobServiceClient(
    env_variable: Optional[str] = "AZURE_STORAGE_CONNECTION_STRING",
) -> BlobServiceClient:
    """
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
    """

    connect_str = os.getenv(env_variable)
    if connect_str is None or connect_str in [""]:
        print(
            f"{env_variable} environment variable is not created!\n"
            f"Pleases follow this link: {_LINK_SETUP_AZURE}\n"
            f"Check the environment variable by (from terminal): 'echo ${env_variable}'"  # noqa
        )
        return

    # --- Initialize
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # --- Print list of containers
    print("List of containers:")
    for blob in blob_service_client.list_containers():
        print("\t" + blob.name)

    return blob_service_client


# -------- initContainerClient
def initContainerClient(
    blob_client: BlobClient, container: str
) -> Tuple[ContainerClient, List]:
    """
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
    """

    # Get the container client
    container_client = blob_client.get_container_client(container)

    # List all blobs in that container
    list_containers = list(container_client.list_blobs())

    print(f"Container initialized. Number of blobs: {len(list_containers)}")

    return container_client, list_containers


def downloadClient(
    container_client: ContainerClient,
    list_containers: List,
    download_dir: str,
    include_dirnames: Optional[Union[str, List[str]]] = None,
    nls_names: Optional[Union[str, List[str]]] = None,
    max_download: Optional[int] = None,
    index1: Optional[int] = 0,
    index2: Optional[int] = None,
    force_download: Optional[bool] = False,
) -> None:
    """
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
    """

    if not os.path.isdir(download_dir):
        os.makedirs(download_dir)

    # Make sure that dirnames and nls_names are of type list
    if include_dirnames and type(include_dirnames) == str:
        include_dirnames = [include_dirnames]
    if nls_names and type(nls_names) == str:
        nls_names = [nls_names]

    # Download files from index1:len(list_containers)
    if index2 is None:
        index2 = len(list_containers)

    divider = 10 * "==="
    counter = 0
    for idx in range(index1, index2):
        blob_filename = list_containers[idx]["name"]
        blob_dirname = os.path.dirname(blob_filename)
        # skip if include_dirnames is specified
        if include_dirnames:
            if blob_dirname not in include_dirnames:
                continue
        if nls_names:
            nls_style_basename = os.path.basename(blob_filename).split(".")[0]
            if nls_style_basename not in nls_names:
                continue

        path2save = os.path.join(download_dir, os.path.basename(blob_filename))

        if os.path.isfile(path2save) and not force_download:
            print(f"{path2save} already exists, skip!")
            continue

        print(divider)
        print(
            f"Download {os.path.basename(list_containers[idx]['name'])} and save it in {path2save}"  # noqa
        )
        one_file_dl = container_client.download_blob(blob_filename)

        with open(path2save, "wb") as download_file:
            one_file_dl.readinto(download_file)
            counter += 1

        if counter >= max_download:
            break

    print(divider)
    print(f"Download finished. Number of new files: {counter}")


def dirnamesBlobs(list_containers: List[Any]) -> List[str]:
    """
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
    """
    all_dirs = []
    print("List of directories detected from the path:")

    for one_container in list_containers:
        dir_name = os.path.dirname(one_container["name"])

        if dir_name not in all_dirs:
            print(f"---> {dir_name}")
            all_dirs.append(dir_name)
