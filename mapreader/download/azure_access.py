#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
try:
    from azure.storage.blob import BlobServiceClient
except ImportError:
    raise ImportError("Please install 'azure-storage-blob'.\n"
                      "Link: https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python#install-the-package")


# Keep this outside of the function so it is easier to change/maintain
LINK_SETUP_AZURE = "https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python#configure-your-storage-connection-string"

# -------- initBlobServiceClient
def initBlobServiceClient(env_variable='AZURE_STORAGE_CONNECTION_STRING'):
    """Instantiate BlobServiceClient

    Note: $AZURE_STORAGE_CONNECTION_STRING needs to be set. 
    """
    
    connect_str = os.getenv(env_variable)
    if connect_str == None or connect_str in [""]:
        print(f"{env_variable} environment variable is not created!\n"
              f"Pleases follow this link: {LINK_SETUP_AZURE}\n"
              f"Check the environment variable by (from terminal): 'echo ${env_variable}'")
        return

    # --- Initialize
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    
    # --- Print list of containers
    print(f"List of containers:")
    for blob in blob_service_client.list_containers():
        print("\t" + blob.name)
        
    return blob_service_client

# -------- initContainerClient
def initContainerClient(blob_client, container):
    """Instantiate a container client

    Arguments:
        blob_client -- Blob Storage Client initialized by initBlobServiceClient function
        container {str} -- Container to access/download from.
    """
    # Get the container client
    container_client = blob_client.get_container_client(container)
    # List all blobs in that container
    list_containers = list(container_client.list_blobs())
    print(f"Container initialized. Number of blobs: {len(list_containers)}")
    return container_client, list_containers

# -------- downloadClient
def downloadClient(container_client, list_containers, download_dir, 
                   include_dirnames=None, nls_names=None,
                   max_download=None,
                   index1=0, index2=None, force_download=False):
    """Download client using a client container

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
    """
    
    if not os.path.isdir(download_dir):
        os.makedirs(download_dir)
    
    # make sure that dirnames and nls_names are of type list
    if include_dirnames and type(include_dirnames) == str:
        include_dirnames = [include_dirnames]
    if nls_names and type(nls_names) == str:
        nls_names = [nls_names]

    # Download files from index1:len(list_containers)
    if index2 == None:
        index2 = len(list_containers)

    counter = 0
    for one_id in range(index1, index2):
        blob_dirname = os.path.dirname(list_containers[one_id]["name"])
        # skip if include_dirnames is specified
        if include_dirnames:
            if not blob_dirname in include_dirnames:
                continue
        if nls_names:
            nls_style_basename = os.path.basename(list_containers[one_id]["name"]).split(".")[0]
            if not nls_style_basename in nls_names:
                continue

        path2save = os.path.join(download_dir, os.path.basename(list_containers[one_id]["name"]))
        
        if os.path.isfile(path2save) and not force_download:
            print(f"{path2save} already exists, skip!")
            continue
            
        print(10*"===")
        print(f"Download {os.path.basename(list_containers[one_id]['name'])} and save it in {path2save}")
        one_file_dl = container_client.download_blob(list_containers[one_id]["name"])

        with open(path2save, "wb") as download_file:
            one_file_dl.readinto(download_file)
            counter += 1
        
        if counter >= max_download:
            break

    print(10*"===")
    print(f"Download finished. Number of new files: {counter}")

# -------- dirnamesBlobs
def dirnamesBlobs(list_containers):
    """Print all dirnames in list_containers"""
    all_dirs = []
    print("List of directories detected from the path:")
    for one_container in list_containers:
        dir_name = os.path.dirname(one_container["name"])
        if not dir_name in all_dirs:
            print(f"---> {dir_name}")
            all_dirs.append(dir_name)
