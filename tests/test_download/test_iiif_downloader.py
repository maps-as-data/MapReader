from __future__ import annotations

import json
import os
from pathlib import Path

import piffle.load_iiif
import pytest
from piffle.iiif_dataclasses.presentation2 import IIIFPresentation2
from piffle.iiif_dataclasses.presentation3 import IIIFPresentation3
from piffle.load_iiif import load_iiif_presentation
from piffle.utils import format_manifest
from PIL import Image

from mapreader import IIIFDownloader


@pytest.fixture(scope="function")
def mock_get_manifest(monkeypatch, sample_dir):
    def mock_get_manifest(url):
        url = sample_dir / url
        with open(url) as f:
            return json.load(f, object_hook=format_manifest)

    monkeypatch.setattr(piffle.load_iiif, "get_manifest", mock_get_manifest)


@pytest.fixture(scope="function")
def mock_download_image(monkeypatch):
    def mock_download_image(self, *args, **kwargs):
        return Image.new("RGB", (10, 10), color=1)

    monkeypatch.setattr(IIIFDownloader, "download_image", mock_download_image)


@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent.parent / "sample_files" / "IIIF_sample_files"


@pytest.fixture
def downloader_str():
    files = ["annotationpage3.json", "annotation3.json", "manifest2.json"]
    uris = ["https://annotations.allmaps.org/manifests/a0d6d3379cfd9f0a", None, None]
    return IIIFDownloader(files, iiif_uri=uris)


@pytest.fixture
def downloader_obj(mock_get_manifest):
    files = ["annotationpage3.json", "annotation3.json", "manifest2.json"]
    uris = ["https://annotations.allmaps.org/manifests/a0d6d3379cfd9f0a", None, None]
    files = [
        load_iiif_presentation(file, str(file).removesuffix(".json")[-1])
        for file in files
    ]
    return IIIFDownloader(files, iiif_uri=uris)


def test_init_str():
    downloader = IIIFDownloader("annotationpage3.json")
    assert isinstance(downloader, IIIFDownloader)
    assert isinstance(downloader.iiif, list)


def test_init_iiif3(mock_get_manifest):
    iiif3 = load_iiif_presentation("annotationpage3.json", 3)
    downloader = IIIFDownloader(iiif3)
    assert isinstance(downloader, IIIFDownloader)
    assert isinstance(downloader.iiif, list)
    assert isinstance(downloader.iiif[0], IIIFPresentation3)


def test_init_iiif2(mock_get_manifest):
    iiif2 = load_iiif_presentation("manifest2.json", 2)
    downloader = IIIFDownloader(iiif2)
    assert isinstance(downloader, IIIFDownloader)
    assert isinstance(downloader.iiif, list)
    assert isinstance(downloader.iiif[0], IIIFPresentation2)


def test_init_list():
    downloader = IIIFDownloader(["annotationpage3.json", "manifest2.json"])
    assert isinstance(downloader, IIIFDownloader)
    assert isinstance(downloader.iiif, list)


def test_save_georeferenced_maps_obj(tmp_path, mock_get_manifest, mock_download_image):
    files = ["annotationpage3.json", "annotation3.json"]
    downloader = IIIFDownloader(
        files,
        iiif_uri=["https://annotations.allmaps.org/manifests/a0d6d3379cfd9f0a", None],
    )
    downloader.save_georeferenced_maps(
        path_save=tmp_path,
    )
    assert os.path.exists(tmp_path / "metadata.csv")
    assert set(
        "5cf13f6681d355e3_masked.tif",
        "5cf13f6681d355e3.tif",
        "bb4029969eeff948.tif",
        "bb4029969eeff948_masked.tif",
        "metadata.csv",
    ) == set(os.listdir(tmp_path))
