# iiifclient
# -*- coding: utf-8 -*-

from collections import OrderedDict
from urllib.parse import urlparse

from cached_property import cached_property
import requests


class IIIFImageClientException(Exception):
    """IIIFImageClient custom exception class"""


class ParseError(IIIFImageClientException):
    """Exception raised when an IIIF image could not be parsed"""


# NOTE: possible image component base class?
# commonalities so far: setting defaults on init / parse
# handling exact matches like full/square? (but maybe only region/size),
# validating options (and could add option type checking)


class ImageRegion(object):
    """IIIF Image region. Intended to be used with :class:`IIIFImageClient`.
    Can be initialized with related image object and region options.

    When associated with an image, region is callable and will return
    an updated image object with the modified region options.

    :param img: :class:`IIFImageClient`
    :param full: full region, defaults to true
    :param square: square region, defaults to false
    :param x: x coordinate
    :param y: y coordinate
    :param width: region width
    :param height: region height
    :param percent: region is a percentage
    """

    # region options
    options = OrderedDict(
        [
            ("full", False),
            ("square", False),
            ("x", None),
            ("y", None),
            ("width", None),
            ("height", None),
            ("percent", False),
        ]
    )

    region_defaults = {
        "full": True,
        "square": False,
        "x": None,
        "y": None,
        "width": None,
        "height": None,
        "percent": False,
    }

    coords = ["x", "y", "width", "height"]

    def __init__(self, img=None, **options):
        self.img = img
        self.options = self.region_defaults.copy()
        if options:
            self.set_options(**options)

    def __call__(self, **options):
        if self.img is not None:
            img = self.img.get_copy()
            img.region.set_options(**options)
            return img

    def set_options(self, **options):
        """Update region options.  Same parameters as initialiation."""
        allowed_options = list(self.options.keys())
        # error if an unrecoganized option is specified
        for key in options:
            if key not in allowed_options:
                raise IIIFImageClientException("Unknown option: %s" % key)

        # error if some but not all coordinates are specified
        # or if percentage is specified but not all coordinates are present
        if (
            any([coord in options for coord in self.coords]) or "percent" in options
        ) and not all([coord in options for coord in self.coords]):
            # partial region specified
            raise IIIFImageClientException("Incomplete region specified")

        # TODO: do we need to type checking? bool/int/float?

        self.options.update(**options)
        # if any non-full value is specified, set full to false
        # NOTE: if e.g. square is specified but false, this is wrong
        allowed_options.remove("full")
        if any([(key in allowed_options and options[key]) for key in options.keys()]):
            self.options["full"] = False

    def as_dict(self):
        """Return region options as a dictionary"""
        return self.options

    def __str__(self):
        """Render region information in IIIF region format"""
        if self.options["full"]:
            return "full"
        if self.options["square"]:
            return "square"

        coords = "%(x)g,%(y)g,%(width)g,%(height)g" % self.options
        if self.options["percent"]:
            return "pct:%s" % coords

        return coords

    def parse(self, region):
        """Parse an IIIF Image region string and update the current region"""

        # reset to defaults before parsing
        self.options = self.region_defaults.copy()

        # full?
        if region == "full":
            self.options["full"] = True
            # return immediately
            return

        self.options["full"] = False

        if region == "square":
            self.options["square"] = True
            # return immediately
            return self

        # percent?
        if "pct" in region:
            self.options["percent"] = True
            region = region.split("pct:")[1]

        # split to dictionary
        # if percentage type, cast to float
        try:
            if self.options["percent"]:
                coords = [float(region_c) for region_c in region.split(",")]
            # else, force int
            else:
                coords = [int(region_c) for region_c in region.split(",")]
        except ValueError:
            # failure converting to integer or float
            raise ParseError("Invalid region coordinates: %s" % region)

        if len(coords) != 4:
            raise ParseError("Invalid region coordinates: %s" % region)

        x, y, width, height = coords
        self.options.update({"x": x, "y": y, "width": width, "height": height})

    def canonicalize(self):
        """Canonicalize the current region options so that
        serialization results in canonical format."""
        # From the spec:
        #   “full” if the whole image is requested, (including a “square”
        #    region of a square image), otherwise the x,y,w,h syntax.

        if self.options["full"]:
            # nothing to do
            return

        # if already in x,y,w,h format - nothing to do
        if not any(
            [self.options["full"], self.options["square"], self.options["percent"]]
        ):
            return

        # possbly an error here for every other case if self.img is not set,
        # since it's probably not possible to canonicalize without knowing
        # image size  (any exceptions?)
        if self.img is None:
            raise IIIFImageClientException("Cannot canonicalize without image")

        if self.options["square"]:
            # if image is square, then return full
            if self.img is not None and self.img.image_width == self.img.image_height:
                self.options["full"] = True
                self.options["square"] = False
                return

            # otherwise convert to x,y,w,h
            # from the spec:
            #  The region is defined as an area where the width and height
            #  are both equal to the length of the shorter dimension of the
            #  complete image. The region may be positioned anywhere in the
            #  longer dimension of the image content at the server’s
            #  discretion, and centered is often a reasonable default.

            # determine size of the short edge
            short_edge = min(self.img.image_width, self.img.image_height)
            width = height = short_edge
            # calculate starting long edge point to center the square
            if self.img.image_height == short_edge:
                y = 0
                x = (self.img.image_width - short_edge) / 2
            else:
                x = 0
                y = (self.img.image_height - short_edge) / 2

            self.options.update(
                {"x": x, "y": y, "width": width, "height": height, "square": False}
            )

        if self.options["percent"]:
            # convert percentages to x,y,w,h
            # From the spec:
            #  The region to be returned is specified as a sequence of
            #  percentages of the full image’s dimensions, as reported in
            #  the image information document. Thus, x represents the number
            #  of pixels from the 0 position on the horizontal axis, calculated
            #  as a percentage of the reported width. w represents the width
            #  of the region, also calculated as a percentage of the reported
            #  width. The same applies to y and h respectively. These may be
            #  floating point numbers.

            # convert percentages to dimensions based on image size
            self.options.update(
                {
                    "percent": False,
                    "x": int((self.options["x"] / 100) * self.img.image_width),
                    "y": int((self.options["y"] / 100) * self.img.image_height),
                    "width": int((self.options["width"] / 100) * self.img.image_width),
                    "height": int(
                        (self.options["height"] / 100) * self.img.image_height
                    ),
                }
            )
            return


class ImageSize(object):
    """IIIF Image Size.  Intended to be used with :class:`IIIFImageClient`.
    Can be initialized with related image object and size options.

    When associated with an image, size is callable and will return
    an updated image object with the modified size.

    :param img: :class:`IIFImageClient`
    :param width: optional width
    :param height: optional height
    :param percent: optional percent
    :param exact: size should be exact (boolean, optional)
    """

    # size options
    options = OrderedDict(
        [
            ("full", False),
            ("max", False),
            ("width", None),
            ("height", None),
            ("percent", None),
            ("exact", False),
        ]
    )

    # NOTE: full is being deprecated and replaced with max;
    # full is deprecated in 2.1 and will be removed for 3.0
    # Eventually piffle will need to address that, maybe with some kind of
    # support for selecting a particular version of the IIIF image spec.
    # For now, default size is still full, and max and full are treated as
    # separate modes.  A parsed url with max will return max, and a parsed
    # url with full will return full, but that will probably change
    # once the deprecated full is handled properly.

    size_defaults = {
        "full": True,
        "max": False,
        "width": None,
        "height": None,
        "percent": None,
        "exact": False,
    }

    def __init__(self, img=None, **options):
        self.img = img
        self.options = self.size_defaults.copy()
        if options:
            self.set_options(**options)

    def __call__(self, **options):
        if self.img is not None:
            img = self.img.get_copy()
            img.size.set_options(**options)
            return img

    def set_options(self, **options):
        """Update size options.  Same parameters as initialiation."""
        allowed_options = list(self.options.keys())
        # error if an unrecoganized option is specified
        for key in options:
            if key not in allowed_options:
                raise IIIFImageClientException("Unknown option: %s" % key)

        # TODO: do we need to type checking? bool/int/float?

        self.options.update(**options)
        # if any non-full value is specified, set full to false
        # NOTE: if e.g. square is specified but false, this is wrong
        allowed_options.remove("full")
        if any([key in allowed_options and options[key] for key in options.keys()]):
            self.options["full"] = False

    def as_dict(self):
        """Return size options as a dictionary"""
        return self.options

    def __str__(self):
        if self.options["full"]:
            return "full"
        if self.options["max"]:
            return "max"
        if self.options["percent"]:
            return "pct:%g" % self.options["percent"]

        size = "%s,%s" % (self.options["width"] or "", self.options["height"] or "")
        if self.options["exact"]:
            return "!%s" % size
        return size

    def parse(self, size):
        # reset to defaults before parsing
        self.options = self.size_defaults.copy()

        # full?
        if size == "full":
            self.options["full"] = True
            return

        # for any other case, full should be false
        self.options["full"] = False

        # max?
        if size == "max":
            self.options["max"] = True
            return

        # percent?
        if "pct" in size:
            try:
                self.options["percent"] = float(size.split(":")[1])
                return
            except ValueError:
                raise ParseError("Error parsing size: %s" % size)

        # exact?
        if size.startswith("!"):
            self.options["exact"] = True
            size = size.lstrip("!")

        # split width and height
        width, height = size.split(",")
        try:
            if width != "":
                self.options["width"] = int(width)
            if height != "":
                self.options["height"] = int(height)
        except ValueError:
            raise ParseError("Error parsing size: %s" % size)

    def canonicalize(self):
        """Canonicalize the current size options so that
        serialization results in canonical format."""
        # From the spec:
        #   “full” if the default size is requested,
        #   the w, syntax for images that should be scaled maintaining the
        #   aspect ratio, and the w,h syntax for explicit sizes that change
        #   the aspect ratio.
        #   Note: The size keyword “full” will be replaced with “max” in
        #   version 3.0

        if self.options["full"]:
            # nothing to do
            return

        # possbly an error here for every other case if self.img is not set,
        # since it's probably not possible to canonicalize without knowing
        # image size  (any exceptions?)
        if self.img is None:
            raise IIIFImageClientException("Cannot canonicalize without image")

        if self.options["percent"]:
            # convert percentage to w,h
            scale = self.options["percent"] / 100
            self.options.update(
                {
                    "height": int(self.img.image_height * scale),
                    "width": int(self.img.image_width * scale),
                    "percent": None,
                }
            )
            return

        if self.options["exact"]:
            # from the spec:
            #   The image content is scaled for the best fit such that the
            #   resulting width and height are less than or equal to the
            #   requested width and height. The exact scaling may be determined
            #   by the service provider, based on characteristics including
            #   image quality and system performance. The dimensions of the
            #   returned image content are calculated to maintain the aspect
            #   ratio of the extracted region.

            # determine which edge results in a smaller scale
            wscale = float(self.options["width"]) / float(self.img.image_width)
            hscale = float(self.options["height"]) / float(self.img.image_height)
            scale = min(wscale, hscale)
            # use that scale on original image size, to preserve
            # original aspect ratio
            self.options.update(
                {
                    "exact": False,
                    "height": int(scale * self.img.image_height),
                    "width": int(scale * self.img.image_width),
                }
            )
            return

        # if height only is specified (,h), convert to width only (w,)
        if self.options["height"] and self.options["width"] is None:
            # determine the scale in use, and convert from
            # height only to width only
            scale = float(self.options["height"]) / float(self.img.image_height)
            self.options.update(
                {"height": None, "width": int(scale * self.img.image_width)}
            )


class ImageRotation(object):
    """IIIF Image rotation Intended to be used with :class:`IIIFImageClient`.
    Can be initialized with related image object and rotation options.

    When associated with an image, rotation is callable and will return
    an updated image object with the modified rotatoin options.

    :param img: :class:`IIFImageClient`
    :param degrees: degrees rotation, optional
    :param mirrored: image should be mirrored (boolean, optional, default
       is False)
    """

    # rotation options
    options = OrderedDict(
        [
            ("degrees", None),
            ("mirrored", False),
        ]
    )

    rotation_defaults = {"degrees": 0, "mirrored": False}

    def __init__(self, img=None, **options):
        self.img = img
        self.options = self.rotation_defaults.copy()
        if options:
            self.set_options(**options)

    def __call__(self, **options):
        if self.img is not None:
            img = self.img.get_copy()
            img.rotation.set_options(**options)
            return img

    def set_options(self, **options):
        """Update size options.  Same parameters as initialiation."""
        allowed_options = self.options.keys()
        # error if an unrecoganized option is specified
        for key in options:
            if key not in allowed_options:
                raise IIIFImageClientException("Unknown option: %s" % key)

        # TODO: do we need to type checking? bool/int/float?

        self.options.update(**options)

    def as_dict(self):
        """Return rotation options as a dictionary"""
        return self.options

    def __str__(self):
        return "%s%g" % (
            "!" if self.options["mirrored"] else "",
            self.options["degrees"],
        )

    def parse(self, rotation):
        # reset to defaults before parsing
        self.options = self.rotation_defaults.copy()

        if str(rotation).startswith("!"):
            self.options["mirrored"] = True
            rotation = rotation.lstrip("!")

        # rotation allows float
        self.options["degrees"] = float(rotation)

    def canonicalize(self):
        """Canonicalize the current region options so that
        serialization results in canonical format."""
        # NOTE: explicitly including a canonicalize as a method to make it
        # clear that this field supports canonicalization, but no work
        # is needed since the existing render does the right things:
        # - trim any trailing zeros in a decimal value
        # - leading zero if less than 1
        # - ! if mirrored, followed by integer if possible
        return


class IIIFImageClient(object):
    """Simple IIIF Image API client for generating IIIF image urls
     in an object-oriented, pythonic fashion.  Can be extended,
     when custom logic is needed to set the image id.  Provides
     a fluid interface, so that IIIF methods can be chained, e.g.::

         iiif_img.size(width=300).rotation(90).format('png')

    Note that this returns a new image instance with the specified
    options, and the original image will remain unchanged.

     .. Note::

         Method to set quality not yet available.
    """

    api_endpoint = None
    image_id = None
    default_format = "jpg"

    # iiif defaults for each sections
    image_defaults = {
        "quality": "default",  # color, gray, bitonal, default
        "fmt": default_format,
    }
    allowed_formats = ["jpg", "tif", "png", "gif", "jp2", "pdf", "webp"]

    def __init__(
        self,
        api_endpoint=None,
        image_id=None,
        region=None,
        size=None,
        rotation=None,
        quality=None,
        fmt=None,
    ):
        self.image_options = self.image_defaults.copy()
        # NOTE: using underscore to differenteate objects from methods
        # but it could be reasonable to make objects public
        self.region = ImageRegion(self)
        self.size = ImageSize(self)
        self.rotation = ImageRotation(self)

        if api_endpoint is not None:
            # remove any trailing slash to avoid duplicate slashes
            self.api_endpoint = api_endpoint.rstrip("/")

        # FIXME: image_id is not required on init to allow subclassing
        # and customizing via get_image_id, but should probably cause
        # an error if you attempt to serialize the url and it is not set
        # (same for a few other options, probably, too...)
        if image_id is not None:
            self.image_id = image_id

        # for now, if region option is specified parse as string
        if region is not None:
            self.region.parse(region)
        if size is not None:
            self.size.parse(size)
        if rotation is not None:
            self.rotation.parse(rotation)

        if quality is not None:
            self.image_options["quality"] = quality
        if fmt is not None:
            self.image_options["fmt"] = fmt

    def get_image_id(self):
        "Image id to be used in contructing urls"
        return self.image_id

    def __str__(self):
        info = self.image_options.copy()
        info.update(
            {
                "endpoint": self.api_endpoint,
                "id": self.get_image_id(),
                "region": str(self.region),
                "size": str(self.size),
                "rot": str(self.rotation),
            }
        )
        return (
            "%(endpoint)s/%(id)s/%(region)s/%(size)s/%(rot)s/%(quality)s.%(fmt)s" % info
        )

    def __repr__(self):
        return "<IIIFImageClient %s>" % self.get_image_id()
        # include non-defaults?

    def info(self):
        "JSON info url"
        return "%(endpoint)s/%(id)s/info.json" % {
            "endpoint": self.api_endpoint,
            "id": self.get_image_id(),
        }

    @cached_property
    def image_info(self):
        "Retrieve image information provided as JSON at info url"
        resp = requests.get(self.info())
        if resp.status_code == requests.codes.ok:
            return resp.json()

        resp.raise_for_status()

    @property
    def image_width(self):
        "Image width as reported in :attr:`image_info`"
        return self.image_info["width"]

    @property
    def image_height(self):
        "Image height as reported in :attr:`image_info`"
        return self.image_info["height"]

    def get_copy(self):
        "Get a clone of the current settings for modification."
        clone = self.__class__(self.api_endpoint, self.image_id, **self.image_options)
        # copy region, size, and rotation - no longer included in
        # image_options dict
        clone.region.set_options(**self.region.as_dict())
        clone.size.set_options(**self.size.as_dict())
        clone.rotation.set_options(**self.rotation.as_dict())
        return clone

    # method to set quality not yet implemented

    def format(self, image_format):
        "Set output image format"
        if image_format not in self.allowed_formats:
            raise IIIFImageClientException("Image format %s unknown" % image_format)
        img = self.get_copy()
        img.image_options["fmt"] = image_format
        return img

    def canonicalize(self):
        """Canonicalize the URI"""
        img = self.get_copy()
        img.region.canonicalize()
        img.size.canonicalize()
        img.rotation.canonicalize()
        return img

    @classmethod
    def init_from_url(cls, url):
        """Init ImageClient using Image API parameters from URI.  Detect
        image vs. info request. Can count reliably from the end of the URI
        backwards, but cannot assume how many slashes make up the api_endpoint.
        Returns new instance of IIIFImageClient.
        Per http://iiif.io/api/image/2.0/#image-request-uri-syntax, using
        slashes to parse URI"""

        # first parse as a url
        parsed_url = urlparse(url)
        # then split the path on slashes
        # and remove any empty strings
        path_components = [path for path in parsed_url.path.split("/") if path]
        if not path_components:
            raise ParseError("Invalid IIIF image url: %s" % url)
        # pop off last portion of the url to determine if this is an info url
        path_basename = path_components.pop()
        opts = {}

        # info request
        if path_basename == "info.json":
            # NOTE: this is unlikely to happen; more likely, if information is
            # missing, we will misinterpret the api endpoint or the image id
            if len(path_components) < 1:
                raise ParseError("Invalid IIIF image information url: %s" % url)
            image_id = path_components.pop()

        # image request
        else:
            # check for enough IIIF parameters
            if len(path_components) < 4:
                raise ParseError("Invalid IIIF image request: %s" % url)

            # pop off url portions as they are used so we can easily
            # make use of leftover path to reconstruct the api endpoint
            quality, fmt = path_basename.split(".")
            rotation = path_components.pop()
            size = path_components.pop()
            region = path_components.pop()
            image_id = path_components.pop()
            opts.update(
                {
                    "region": region,
                    "size": size,
                    "rotation": rotation,
                    "quality": quality,
                    "fmt": fmt,
                }
            )

        # construct the api endpoint url from the parsed url and whatever
        # portions of the url path are leftover
        # remove empty strings from the remaining path components
        path_components = [p for p in path_components if p]
        api_endpoint = "%s://%s/%s" % (
            parsed_url.scheme,
            parsed_url.netloc,
            "/".join(path_components) if path_components else "",
        )

        # init and return instance
        return cls(api_endpoint=api_endpoint, image_id=image_id, **opts)

    def as_dict(self):
        """
        Dictionary of with all image request options.
        request parameters. Returns a dictionary with all image request
        parameters parsed to their most granular level. Can be helpful
        for acting logically on particular request parameters like height,
        width, mirroring, etc.
        """
        return OrderedDict(
            [
                ("region", self.region.as_dict()),
                ("size", self.size.as_dict()),
                ("rotation", self.rotation.as_dict()),
                ("quality", self.image_options["quality"]),
                ("format", self.image_options["fmt"]),
            ]
        )
