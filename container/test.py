#!/usr/bin/env python3

"""
run_tests.py

A Python script to automate the building of the Docker image, running the container,
executing dependency tests for MapReader, and reporting the results.

Usage:
    python3 run_tests.py

Requirements:
    - Python 3.6+
    - podman-hpc installed and configured
    - Dockerfile and requirements.txt present in the current directory
"""

import subprocess
import sys
import os
from typing import List, Tuple

# Configuration Variables
IMAGE_NAME = "mapreader:latest"
CONTAINER_NAME = "mapreader"
WORKDIR = "/workspace"

# Define the dependency tests as a list of tuples (Test Description, Command)
# Each dependency has two commands: one to print the library name and another to print its version
DEPENDENCY_TESTS: List[Tuple[str, str]] = [
    ("PyTorch", "python3 -c 'import torch; print(torch.__version__)'"),
    ("PyTorch GPU", "python3 -c 'import torch; print(torch.cuda.is_available())'"),
    ("Torchvision", "python3 -c 'import torchvision; print(torchvision.__version__)'"),
    ("Geopandas", "python3 -c 'import geopandas; print(geopandas.__version__)'"),
    ("Geopy", "python3 -c 'import geopy; print(geopy.__version__)'"),
    ("Cython Python", "python3 -c 'import Cython; print(Cython.__version__)'"),
    ("Torchinfo", "python3 -c 'import torchinfo; print(torchinfo.__version__)'"),
    ("Parhugin MultiFun", "python3 -c 'from parhugin import multiFunc; myproc = multiFunc(num_req_p=10)'"),
    ("MapReader", "python3 -c 'import mapreader; print(mapreader.__version__)'"),
    ("GDAL", "gdalinfo --version"),
    ("Fiona", "python3 -c 'import fiona; print(fiona.__version__)'"),
    ("Shapely", "python3 -c 'import shapely; print(shapely.__version__)'"),
    ("Scikit-learn", "python3 -c 'import sklearn; print(sklearn.__version__)'"),
    ("Scikit-image", "python3 -c 'import skimage; print(skimage.__version__)'"),
    ("Tensorboard", "python3 -c 'import tensorboard; print(tensorboard.__version__)'"),
    ("Jupyter", "jupyter --version"),
    ("IPython Kernel", "python3 -c 'import ipykernel; print(ipykernel.__version__)'"),
    ("IPyWidgets", "python3 -c 'import ipywidgets; print(ipywidgets.__version__)'"),
    ("OpenCV", "python3 -c 'import cv2; print(cv2.__version__)'"),
    ("Rasterio", "python3 -c 'import rasterio; print(rasterio.__version__)'"),
]

def check_file_exists(filename: str) -> bool:
    """
    Check if a file exists in the current directory.
    """
    exists = os.path.isfile(filename)
    if not exists:
        print(f"❌ {filename} not found in the current directory.")
    return exists

def run_command(command: str, capture_output: bool = True) -> subprocess.CompletedProcess:
    """
    Run a shell command and return the CompletedProcess object.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {command}")
        if capture_output:
            print(f"--- STDOUT ---\n{e.stdout}")
            print(f"--- STDERR ---\n{e.stderr}")
        raise

def build_image():
    """
    Build the Docker image using podman-hpc.
    """
    print(f"=== Building Docker Image: {IMAGE_NAME} ===")
    build_command = f"podman-hpc build -t {IMAGE_NAME} ."
    run_command(build_command)
    print(f"✅ Docker Image '{IMAGE_NAME}' built successfully.\n")

def run_container() -> str:
    """
    Run the container and return the container ID.
    """
    print(f"=== Running Container: {CONTAINER_NAME} ===")
    run_command(
        f"podman-hpc run -d --gpu --name {CONTAINER_NAME} "
        f"{IMAGE_NAME} sleep infinity"
    )
    # Get container ID
    container_id = run_command(f"podman-hpc inspect -f '{{{{.Id}}}}' {CONTAINER_NAME}").stdout.strip()
    print(f"✅ Container '{CONTAINER_NAME}' is running with ID: {container_id}\n")
    return container_id

def execute_tests(container_id: str) -> List[Tuple[str, bool, str]]:
    """
    Execute dependency tests inside the container and return the results.

    Returns:
        List of tuples containing (Test Description, Success (True/False), Output)
    """
    print(f"=== Executing Dependency Tests Inside Container '{CONTAINER_NAME}' ===\n")
    results = []

    for test_desc, test_cmd in DEPENDENCY_TESTS:
        print(f"🔍 Running Test: {test_desc}")
        
        # Execute the library name command
        name_cmd = f"echo '{test_desc}'"
        try:
            name_result = run_command(f"podman-hpc exec {container_id} bash -c \"{name_cmd}\"")
            library_name = name_result.stdout.strip()
            print(f"📦 Library: {library_name}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to retrieve library name for {test_desc}.")
            results.append((test_desc, False, "Failed to retrieve library name."))
            continue  # Skip to the next test
        
        # Execute the version command
        version_cmd = test_cmd
        try:
            version_result = run_command(f"podman-hpc exec {container_id} bash -c \"{version_cmd}\"")
            version_output = version_result.stdout.strip()
            print(f"✅ {library_name} Output: {version_output}\n")
            results.append((library_name, True, version_output))
        except subprocess.CalledProcessError:
            print(f"❌ {library_name} Output: Failed to execute.\n")
            results.append((library_name, False, "Failed to execute."))
    
    return results

def analyze_results(results: List[Tuple[str, bool, str]]) -> bool:
    """
    Analyze the test results and return True if all tests passed, False otherwise.
    """
    all_passed = True
    print(f"=== Analyzing Test Results ===\n")
    for test_desc, success, output in results:
        if success:
            print(f"✅ {test_desc}: {output}")
        else:
            print(f"❌ {test_desc}: {output}")
            all_passed = False
    print("\n=== Test Analysis Complete ===\n")
    return all_passed

def cleanup(container_id: str):
    """
    Stop and remove the container.
    """
    print(f"=== Cleaning Up: Removing Container '{CONTAINER_NAME}' ===")
    try:
        run_command(f"podman-hpc stop {container_id}")
        run_command(f"podman-hpc rm {container_id}")
        print(f"✅ Container '{CONTAINER_NAME}' has been removed.\n")
    except subprocess.CalledProcessError:
        print(f"⚠️  Failed to remove container '{CONTAINER_NAME}'. Please remove it manually.\n")

def main():
    """
    Main function to orchestrate the build, run, test, and cleanup processes.
    """
    # Check for necessary files
    if not check_file_exists("Dockerfile") or not check_file_exists("requirements.txt"):
        sys.exit(1)

    try:
        # Build the Docker image
        # build_image()

        # Run the container and get container ID
        container_id = run_container()

        # Execute dependency tests
        test_results = execute_tests(container_id)

        # Analyze results
        all_passed = analyze_results(test_results)

        if all_passed:
            print("🎉 All dependency tests passed successfully!")
        else:
            print("⚠️  Some dependency tests failed. Please review the above output for details.")
            sys.exit(1)

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        sys.exit(1)

    finally:
        # Cleanup: Stop and remove the container
        if 'container_id' in locals():
            cleanup(container_id)

if __name__ == "__main__":
    main()
