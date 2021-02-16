import json
import os.path
from unittest.mock import patch

import pytest
import requests

from piffle.presentation import IIIFPresentation, IIIFException


FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


class TestIIIFPresentation:
    test_manifest = os.path.join(FIXTURE_DIR, 'chto-manifest.json')

    def test_from_file(self):
        pres = IIIFPresentation.from_file(self.test_manifest)
        assert isinstance(pres, IIIFPresentation)
        assert pres.type == 'sc:Manifest'

    def test_from_url(self):
        manifest_url = 'http://ma.ni/fe.st'
        with open(self.test_manifest) as manifest:
            data = json.loads(manifest.read())
        with patch('piffle.presentation.requests') as mockrequests:
            mockrequests.codes = requests.codes
            mockresponse = mockrequests.get.return_value
            mockresponse.status_code = requests.codes.ok
            mockresponse.json.return_value = data
            pres = IIIFPresentation.from_url(manifest_url)
            assert pres.type == 'sc:Manifest'
            mockrequests.get.assert_called_with(manifest_url)
            mockrequests.get.return_value.json.assert_called_with()

            # error handling
            # bad status code response on the url
            with pytest.raises(IIIFException) as excinfo:
                mockresponse.status_code = requests.codes.forbidden
                mockresponse.reason = 'Forbidden'
                IIIFPresentation.from_url(manifest_url)
            assert 'Error retrieving manifest' in str(excinfo.value)
            assert '403 Forbidden' in str(excinfo.value)

            # valid http response but not a json response
            with pytest.raises(IIIFException) as excinfo:
                mockresponse.status_code = requests.codes.ok
                # content type header does not indicate json
                mockresponse.headers = {'content-type': 'text/html'}
                mockresponse.json.side_effect = \
                    json.decoder.JSONDecodeError('err', 'doc', 1)
                IIIFPresentation.from_url(manifest_url)
            assert 'No JSON found' in str(excinfo.value)

            # json parsing error
            with pytest.raises(IIIFException) as excinfo:
                # content type header indicates json, but parsing failed
                mockresponse.headers = {'content-type': 'application/json'}
                mockresponse.json.side_effect = \
                    json.decoder.JSONDecodeError('err', 'doc', 1)
                IIIFPresentation.from_url(manifest_url)
            assert 'Error parsing JSON' in str(excinfo.value)

    def test_from_url_or_file(self):
        with patch.object(IIIFPresentation, 'from_url') as mock_from_url:
            # local fixture file
            pres = IIIFPresentation.from_file_or_url(self.test_manifest)
            assert pres.type == 'sc:Manifest'
            mock_from_url.assert_not_called()

            pres = IIIFPresentation.from_file_or_url('http://mani.fe/st')
            mock_from_url.assert_called_with('http://mani.fe/st')

            # nonexistent file path
            with pytest.raises(IIIFException) as excinfo:
                IIIFPresentation.from_file_or_url('/manifest/not/found')
            assert 'File not found: ' in str(excinfo.value)

    def test_short_id(self):
        manifest_uri = 'https://ii.if/resources/p0c484h74c/manifest'
        assert IIIFPresentation.short_id(manifest_uri) == 'p0c484h74c'
        canvas_uri = 'https://ii.if/resources/p0c484h74c/manifest/canvas/ps7527b878'
        assert IIIFPresentation.short_id(canvas_uri) == 'ps7527b878'

    def test_toplevel_attrs(self):
        pres = IIIFPresentation.from_file(self.test_manifest)
        assert pres.context == "http://iiif.io/api/presentation/2/context.json"
        assert pres.id == "https://plum.princeton.edu/concern/scanned_resources/ph415q7581/manifest"
        assert pres.type == "sc:Manifest"
        assert pres.label[0] == "Chto my stroim : Tetrad\u02b9 s kartinkami"
        assert pres.viewingHint == "paged"
        assert pres.viewingDirection == "left-to-right"

    def test_nested_attrs(self):
        pres = IIIFPresentation.from_file(self.test_manifest)
        assert isinstance(pres.sequences, tuple)
        assert pres.sequences[0].id == \
            "https://plum.princeton.edu/concern/scanned_resources/ph415q7581/manifest/sequence/normal"
        assert pres.sequences[0].type == "sc:Sequence"
        assert isinstance(pres.sequences[0].canvases, tuple)
        assert pres.sequences[0].canvases[0].id == \
            "https://plum.princeton.edu/concern/scanned_resources/ph415q7581/manifest/canvas/p02871v98d"

    def test_set(self):
        pres = IIIFPresentation.from_file(self.test_manifest)
        pres.label = 'New title'
        pres.type = 'sc:Collection'
        assert pres.label == 'New title'
        assert pres.type == 'sc:Collection'

    def test_del(self):
        pres = IIIFPresentation.from_file(self.test_manifest)
        del pres.label
        del pres.type
        assert not hasattr(pres, 'label')
        assert not hasattr(pres, 'type')

    def test_first_label(self):
        pres = IIIFPresentation.from_file(self.test_manifest)
        assert pres.first_label == pres.label[0]
        pres.label = 'unlisted single title'
        assert pres.first_label == pres.label
