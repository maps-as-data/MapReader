# iiifclient

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
        'format': default_format
    }
    allowed_formats = ['jpg', 'tif', 'png', 'gif', 'jp2', 'pdf', 'webp']

    def __init__(self, api_endpoint=None, image_id=None, region=None,
                 size=None, rotation=None, quality=None, format=None):
        self.image_options = self.image_defaults.copy()
        if api_endpoint is not None:
            self.api_endpoint = api_endpoint
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
        if format is not None:
            self.image_options['format'] = format

    def get_image_id(self):
        'Image id to be used in contructing urls'
        return self.image_id

    def __unicode__(self):
        info = self.image_options.copy()
        info.update({
            'endpoint': self.api_endpoint.rstrip('/'), # avoid duplicate slashes',
            'id': self.get_image_id(),
        })
        return '%(endpoint)s/%(id)s/%(region)s/%(size)s/%(rotation)s/%(quality)s.%(format)s' % info

    def __str__(self):
        return str(unicode(self))

    def __repr__(self):
        return '<IIIFImageClient %s>' % self.get_image_id()
        # include non-defaults?

    def info(self):
        'JSON info url'
        return '%(endpoint)s/%(id)s/info.json' %  {
            'endpoint': self.api_endpoint.rstrip('/'), # avoid duplicate slashes',
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
            raise Exception('Image format %s unknown' % image_format)
        img = self.get_copy()
        img.image_options['format'] = image_format
        return img
