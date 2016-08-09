# iiifclient

from urlparse import urlparse

class IIIFImageClientException(Exception):
    '''IIIFImageClient custom exception class'''
    pass


class URLParseError(IIIFImageClientException):
    '''Exception raised when an IIIF image could not be parsed'''
    pass


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
        'region': 'full',  # full image, no cropping
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
        if region is not None:
            self.image_options['region'] = region
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
                raise URLParseError('Invalid IIIF image information url: %s'
                                    % url)
            image_id = path_components.pop()

        # image request
        else:
            # check for enough IIIF parameters
            if len(path_components) < 4:
                raise URLParseError('Invalid IIIF image request: %s' % url)

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
        print 'path components = ', path_components
        # remove empty strings from the remaining path components
        path_components = [p for p in path_components if p]
        print parsed_url.netloc
        print '/'.join(path_components) if path_components else ''
        print [p for p in path_components if p]
        api_endpoint = '%s://%s/%s' % (
            parsed_url.scheme, parsed_url.netloc,
            '/'.join(path_components) if path_components else '')

        print 'api endpoint =', api_endpoint

        # init and return instance
        return cls(api_endpoint=api_endpoint, image_id=image_id, **opts)

    def dict_opts(self):
        '''
        Aggregate method that fires other client methods that parse image request parameters.
        Return a dictionary with all image request parameters parsed to their most granular level.
        Can be helpful for acting logically on particular request parameters like height,
        width, mirroring, etc.
        '''

        return {
            'region': self.region_as_dict(),
            'size': self.size_as_dict(),
            'rotation': self.rotation_as_dict()
        }

    # methods to derive python dictionaries from IIIF strings
    def region_as_dict(self):
        '''Return region options as a dictionary'''

        # preliminary region options
        region_dict = {
            'full': False,
            'x': None,
            'y': None,
            'w': None,
            'h': None,
            'pct': False
        }

        region = self.image_options['region']

        # full?
        if region == 'full':
            region_dict['full'] = True
            # return immediately
            return region_dict

        # percent?
        if "pct" in region:
            region_dict['pct'] = True
            region = region.split("pct:")[1]

        # split to dictionary
        # if percentage type, cast to float
        if region_dict['pct']:
            x, y, w, h = [float(region_c) for region_c in region.split(",")]
        # else, force int
        else:
            x, y, w, h = [int(region_c) for region_c in region.split(",")]
        region_dict.update({'x': x, 'y': y, 'w': w, 'h': h})

        return region_dict

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
