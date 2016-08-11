# iiifclient

from collections import OrderedDict
from urlparse import urlparse


class IIIFImageClientException(Exception):
    '''IIIFImageClient custom exception class'''
    pass


class ParseError(IIIFImageClientException):
    '''Exception raised when an IIIF image could not be parsed'''
    pass


class ImageRegion(object):
    '''IIIF Image region.  Region options can be specified on
    initialization.

    :param full: full region, defaults to true
    :param square: square region, defaults to false
    :param x: x coordinate
    :param y: y coordinate
    :param width: region width
    :param height: region height
    :param percentage: region is a percentage
    '''

    # region options
    options = OrderedDict([
        ('full', None),
        ('square', False),
        ('x', None),
        ('y', None),
        ('width', None),
        ('height', None),
        ('percentage', False)
    ])

    region_defaults = {
        'full': True,
        'square': False,
        'x': None,
        'y': None,
        'width': None,
        'height': None,
        'percentage': False
    }

    coords = ['x', 'y', 'width', 'height']

    def __init__(self, **options):
        self.options = self.region_defaults.copy()
        if options:
            self.set_options(options)

    def set_options(self, **options):
        allowed_options = self.options.keys()
        # error if an unrecoganized option is specified
        for key in options:
            if key not in allowed_options:
                raise IIIFImageClientException('Unknown option: %s' % key)

        # error if some but not all coordinates are specified
        # or if percentage is specified but not all coordinates are present
        if (any([coord in options for coord in self.coords]) or
           'percentage' in options) and not \
           all([coord in options for coord in self.coords]):
            # partial region specified
            raise IIIFImageClientException('Incomplete region specified')

        # TODO: do we need to type checking? bool/int/float?

        self.options.update(**options)
        # if any non-full value is specified, set full to false
        # NOTE: if e.g. square is specified but false, this is wrong
        allowed_options.remove('full')
        if any([key in allowed_options for key in options.keys()]):
            self.options['full'] = False

    def as_dict(self):
        '''Return region options as a dictionary'''
        return self.options

    def __unicode__(self):
        '''Render region information in IIIF region format'''
        if self.options['full']:
            return 'full'
        if self.options['square']:
            return 'square'
        if self.options['percentage']:
            return 'pct:%(x)g,%(y)g,%(width)g,%(height)g' %  \
                self.options
        else:
            return '%(x)d,%(y)d,%(width)d,%(height)d' % self.options

    def parse(self, region):
        '''Parse an IIIF Image region string'''

        # reset to defaults before parsing
        self.options = self.region_defaults.copy()

        # full?
        if region == 'full':
            self.options['full'] = True
            # return immediately
            return
        else:
            self.options['full'] = False

        if region == 'square':
            self.options['square'] = True
            # return immediately
            return self

        # percent?
        if "pct" in region:
            self.options['percentage'] = True
            region = region.split("pct:")[1]

        # split to dictionary
        # if percentage type, cast to float
        try:
            if self.options['percentage']:
                coords = [float(region_c) for region_c in region.split(",")]
            # else, force int
            else:
                coords = [int(region_c) for region_c in region.split(",")]
        except ValueError:
            # failure converting to integer or float
            raise ParseError('Invalid region coordinates: %s' % region)

        if len(coords) != 4:
            raise ParseError('Invalid region coordinates: %s' % region)

        x, y, width, height = coords
        self.options.update({'x': x, 'y': y, 'width': width, 'height': height})


class IIIFImageClient(object):
    '''Simple IIIF Image API client for generating IIIF image urls
    in an object-oriented, pythonic fashion.  Can be extended,
    when custom logic is needed to set the image id.  Provides
    a fluid interface, so that IIIF methods can be chained, e.g.::

        iiif_img.size(width=300).format('png')

    .. Note::

        Methods to set region, rotation, and quality are not yet
        implemented.
    '''

    api_endpoint = None
    image_id = None
    default_format = 'jpg'

    # iiif defaults for each sections
    image_defaults = {
        'size': 'full',    # full size, unscaled
        'rotation': '0',   # no rotation
        'quality': 'default',  # color, gray, bitonal, default
        'fmt': default_format
    }
    allowed_formats = ['jpg', 'tif', 'png', 'gif', 'jp2', 'pdf', 'webp']

    def __init__(self, api_endpoint=None, image_id=None, region=None,
                 size=None, rotation=None, quality=None, fmt=None):
        self.image_options = self.image_defaults.copy()
        if api_endpoint is not None:
            # remove any trailing slash to avoid duplicate slashes
            self.api_endpoint = api_endpoint.rstrip('/')

        # FIXME: image_id is not required on init to allow subclassing
        # and customizing via get_image_id, but should probably cause
        # an error if you attempt to serialize the url and it is not set
        # (same for a few other options, probably, too...)
        if image_id is not None:
            self.image_id = image_id

        # init region
        self.region = ImageRegion()
        # for now, if region option is specified parse as string
        if region is not None:
            self.region.parse(region)

        if size is not None:
            self.image_options['size'] = size
        if rotation is not None:
            self.image_options['rotation'] = rotation
        if quality is not None:
            self.image_options['quality'] = quality
        if fmt is not None:
            self.image_options['fmt'] = fmt

    def get_image_id(self):
        'Image id to be used in contructing urls'
        return self.image_id

    def __unicode__(self):
        info = self.image_options.copy()
        info.update({
            'endpoint': self.api_endpoint,
            'id': self.get_image_id(),
            'region': unicode(self.region)
        })
        return '%(endpoint)s/%(id)s/%(region)s/%(size)s/%(rotation)s/%(quality)s.%(fmt)s' % info

    def __str__(self):
        return str(unicode(self))

    def __repr__(self):
        return '<IIIFImageClient %s>' % self.get_image_id()
        # include non-defaults?

    def info(self):
        'JSON info url'
        return '%(endpoint)s/%(id)s/info.json' % {
            'endpoint': self.api_endpoint,
            'id': self.get_image_id(),
        }

    def get_copy(self):
        'Get a clone of the current settings for modification.'
        return self.__class__(self.api_endpoint, self.image_id, **self.image_options)

    # methods to set region, rotation, quality not yet implemented

    def size(self, width=None, height=None, percent=None, exact=False):
        '''Set image size.  May specify any one of width, height, or percent,
        or both width and height, optionally specifying best fit / exact
        scaling.'''
        # width only
        if width is not None and height is None:
            size = '%s,' % (width, )
        # height only
        elif height is not None and width is None:
            size = ',%s' % (height, )
        # percent
        elif percent is not None:
            size = 'pct:%s' % (percent, )
        # both width and height
        elif width is not None and height is not None:
            size = '%s,%s' % (width, height)
            if exact:
                size = '!%s' % size

        img = self.get_copy()
        img.image_options['size'] = size
        return img

    def format(self, image_format):
        'Set output image format'
        if image_format not in self.allowed_formats:
            raise IIIFImageClientException('Image format %s unknown' % image_format)
        img = self.get_copy()
        img.image_options['fmt'] = image_format
        return img

    @classmethod
    def init_from_url(cls, url):
        '''Init ImageClient using Image API parameters from URI.  Detect
        image vs. info request. Can count reliably from the end of the URI
        backwards, but cannot assume how many slashes make up the api_endpoint.
        Returns new instance of IIIFImageClient.
        Per http://iiif.io/api/image/2.0/#image-request-uri-syntax, using
        slashes to parse URI'''

        # first parse as a url
        parsed_url = urlparse(url)
        # then split the path on slashes
        path_components = parsed_url.path.split('/')
        # pop off last portion of the url to determine if this is an info url
        path_basename = path_components.pop()
        opts = {}

        # info request
        if path_basename == 'info.json':
            if len(path_components) < 1:
                raise ParseError('Invalid IIIF image information url: %s'
                                    % url)
            image_id = path_components.pop()

        # image request
        else:
            # check for enough IIIF parameters
            if len(path_components) < 4:
                raise ParseError('Invalid IIIF image request: %s' % url)

            # pop off url portions as they are used so we can easily
            # make use of leftover path to reconstruct the api endpoint
            quality, fmt = path_basename.split('.')
            rotation = path_components.pop()
            size = path_components.pop()
            region = path_components.pop()
            image_id = path_components.pop()
            opts.update({
                'region': region,
                'size': size,
                'rotation': rotation,
                'quality': quality,
                'fmt': fmt
            })

        # construct the api endpoint url from the parsed url and whatever
        # portions of the url path are leftover
        # remove empty strings from the remaining path components
        path_components = [p for p in path_components if p]
        api_endpoint = '%s://%s/%s' % (
            parsed_url.scheme, parsed_url.netloc,
            '/'.join(path_components) if path_components else '')

        # init and return instance
        return cls(api_endpoint=api_endpoint, image_id=image_id, **opts)

    def as_dict(self):
        '''
        Aggregate method that fires other client methods that parse image
        request parameters. Returns a dictionary with all image request
        parameters parsed to their most granular level. Can be helpful
        for acting logically on particular request parameters like height,
        width, mirroring, etc.
        '''
        return {
            'region': self.region.as_dict(),
            'size': self.size_as_dict(),
            'rotation': self.rotation_as_dict()
        }

    def size_as_dict(self):
        '''Return size options as a dictionary'''

        # preliminary dictionary
        size_dict = {
            'full': False,
            'w': None,
            'h': None,
            'exact': False,
            'pct': False,
        }

        size = self.image_options['size']

        # full?
        if size == 'full':
            size_dict['full'] = True
            # return immediately
            return size_dict

        # percent?
        if "pct" in size:
            size_dict['pct'] = float(size.split(":")[1])
            return size_dict

        # exact?
        if size.startswith('!'):
            size_dict['exact'] = True
            size = size[1:]

        # split width and height
        w, h = size.split(",")
        if w != '':
            size_dict['w'] = int(w)
        if h != '':
            size_dict['h'] = int(h)

        return size_dict

    def rotation_as_dict(self):
        '''Return rotation options as a dictionary'''
        rotation_dict = {
            'degrees': None,
            'mirrored': False
        }

        rotation = self.image_options['rotation']

        if rotation.startswith('!'):
            rotation_dict['mirrored'] = True
            rotation = rotation[1:]

        # rotation allows float
        rotation_dict['degrees'] = float(rotation)

        return rotation_dict
