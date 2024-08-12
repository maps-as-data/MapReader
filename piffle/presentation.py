import json
import os.path
import urllib

import addict

import requests


class IIIFException(Exception):
    """Custom exception for IIIF errors"""


class AtDict(addict.Dict):
    """Base attrdict class with handling for fields like @type, @id, etc"""

    at_fields = ["type", "id", "context"]

    def _key(self, key):
        # convert key to @key if in the list of fields that requires it
        if key in self.at_fields:
            key = "@%s" % key
        return key

    def __missing__(self, key):
        raise KeyError(self._key(key))

    def __getattr__(self, key):
        try:
            # addict getattr just calls getitem
            return super().__getattr__(self._key(key))
        except KeyError:
            # python hasattr checks for attribute error
            # translate key error to attribute error,
            # since in an attr dict it's kind of both
            raise AttributeError

    def __getitem__(self, key):
        """
        Access a value associated with a key.
        """
        val = super().__getitem__(self._key(key))

        if key == "seeAlso" and isinstance(val, list) and isinstance(val[0], dict):
            return [AtDict(entry) for entry in val]
        return val

    def __setitem__(self, key, value):
        """
        Add a key-value pair to the instance.
        """
        return super().__setitem__(self._key(key), value)

    def __delitem__(self, key):
        """
        Delete a key-value pair
        """
        super().__delitem__(self._key(key))


class IIIFPresentation(AtDict):
    """:class:`addict.Dict` subclass for read access to IIIF Presentation
    content"""

    # TODO: document sample use, e.g. @ fields

    at_fields = ["type", "id", "context"]

    @classmethod
    def get_iiif_url(cls, url):
        """Wrapper around :meth:`requests.get` to support conditionally
        adding an auth tokens or other parameters."""
        request_options = {}
        # TODO: need some way of configuring hooks for e.g. setting auth tokens
        return requests.get(url, **request_options)

    @classmethod
    def from_file(cls, path):
        """Initialize :class:`IIIFPresentation` from a file."""
        with open(path) as manifest:
            data = json.loads(manifest.read())
        return cls(data)

    @classmethod
    def from_url(cls, uri):
        """Iniitialize :class:`IIIFPresentation` from a URL.

        :raises: :class:`IIIFException` if URL is not retrieved successfully,
            if the response is not JSON content, or if the JSON cannot be parsed.
        """
        response = cls.get_iiif_url(uri)
        if response.status_code == requests.codes.ok:
            try:
                return cls(response.json())
            except json.decoder.JSONDecodeError as err:
                # if json fails, two possibilities:
                # - we didn't actually get json (e.g. redirect for auth)
                if "application/json" not in response.headers["content-type"]:
                    raise IIIFException("No JSON found at %s" % uri)
                # - there is something wrong with the json
                raise IIIFException("Error parsing JSON for %s: %s" % (uri, err))

        raise IIIFException(
            "Error retrieving manifest at %s: %s %s"
            % (uri, response.status_code, response.reason)
        )

    @classmethod
    def is_url(cls, url):
        """Utility method to check if a path is a url or file"""
        return urllib.parse.urlparse(url).scheme != ""

    @classmethod
    def from_file_or_url(cls, path):
        """Iniitialize :class:`IIIFPresentation` from a file or a url."""
        if os.path.isfile(path):
            return cls.from_file(path)
        elif cls.is_url(path):
            return cls.from_url(path)
        else:
            raise IIIFException("File not found: %s" % path)

    @classmethod
    def short_id(cls, uri):
        """Generate a short id from full manifest/canvas uri identifiers
        for use in local urls.  Logic is based on the recommended
        url pattern from the IIIF Presentation 2.0 specification."""

        # shortening should work reliably for uris that follow
        # recommended url patterns from the spec
        # http://iiif.io/api/presentation/2.0/#a-summary-of-recommended-uri-patterns
        #   manifest:  {scheme}://{host}/{prefix}/{identifier}/manifest
        #   canvas: {scheme}://{host}/{prefix}/{identifier}/canvas/{name}

        # remove trailing /manifest at the end of the url, if present
        if uri.endswith("/manifest"):
            uri = uri[: -len("/manifest")]
        # split on slashes and return the last portion
        return uri.split("/")[-1]

    @property
    def first_label(self):
        # label can be a string or list of strings
        if isinstance(self.label, str):
            return self.label
        else:
            return self.label[0]
