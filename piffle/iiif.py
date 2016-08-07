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

    @classmethod
    def init_from_url(ic, url):
        '''Init ImageClient using Image API parameters from URI.  Detect image vs. info request.
        Can count reliably from the end of the URI backwards, but cannot assume how many slashes 
        make up the api_endpoint.  Returns new instance of IIIFImageClient.
        Per http://iiif.io/api/image/2.0/#image-request-uri-syntax, using slashes to parse URI'''
        
        # info request
        if url.endswith('info.json'):

            url_components = url.split('/')

            if len(url_components) < 5:
                raise Exception('Not enough IIIF image parameters provided for information request {scheme}://{server}{/prefix}/{identifier}/info.json: %s' % url)

            _image_id = url_components[-2]
            _api_endpoint = '/'.join(url_components[:-3])

            # reinit
            return ic(api_endpoint=_api_endpoint, image_id=_image_id)

        # image request
        else:

            url_components = url.split('/')

            # check for enough IIIF parameters
            if len(url_components) < 8:
                raise Exception('Not enough IIIF image parameters provided for image request {scheme}://{server}{/prefix}/{identifier}/{region}/{size}/{rotation}/{quality}.{format}: %s' % url)
            
            _quality, _format = url_components[-1].split('.')
            _rotation = url_components[-2]
            _size = url_components[-3]
            _region = url_components[-4]
            _image_id = url_components[-5]
            _api_endpoint = '/'.join(url_components[:-6])
            
            # reinit
            return ic(api_endpoint=_api_endpoint, image_id=_image_id, region=_region,
                         size=_size, rotation=_rotation, quality=_quality, format=_format)
                     
    def dict_opts(self):
        
        '''
        Aggregate method that fires other client methods that parse image request parameters.
        Return a dictionary with all image request parameters parsed to their most granular level.
        Can be helpful for acting logically on particular request parameters like height,
        width, mirroring, etc.
        '''

        return {
            'region':self.region_as_dict(),
            'size':self.size_as_dict(),
            'rotation':self.rotation_as_dict()            
        }

    # methods to derive python dictionaries from IIIF strings
    def region_as_dict(self):

        '''Parses region parameter into dictionary'''

        # return dictionary
        region_d = {
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
            region_d['full'] = True
            # return immediately
            return region_d

        # percent?
        if "pct" in region:
            region_d['pct'] = True
            region = region.split("pct:")[1]

        # split to dictionary
        # if percentage type, cast to float
        if region_d['pct']:
            region_d['x'],region_d['y'],region_d['w'],region_d['h'] = [float(region_c) for region_c in region.split(",")]
        # else, force int
        else:
            region_d['x'],region_d['y'],region_d['w'],region_d['h'] = [int(region_c) for region_c in region.split(",")]

        return region_d

    def size_as_dict(self):

        '''Parses size parameter into dictionary'''

        # return dictionary
        size_d = {
        'full': False,
        'w': None,
        'h': None,
        'exact': False,
        'pct': False,
        }

        size = self.image_options['size']

        # full?
        if size == 'full':
            size_d['full'] = True
            # return immediately
            return size_d

        # percent?
        if "pct" in size:
            size_d['pct'] = int(size.split(":")[1])
            return size_d

        # exact?
        if size.startswith('!'):
            size_d['exact'] = True
            size = size[1:]

        # split width and height
        w,h = size.split(",")
        if w != '':
            size_d['w'] = int(w)
        if h != '':
            size_d['h'] = int(h) 

        return size_d

    def rotation_as_dict(self):

        '''Parses rotation parameter into dictionary'''

        rotation_d = {
        'degrees': None,
        'mirrored': False
        }

        rotation = self.image_options['rotation']

        if rotation.startswith('!'):
            rotation_d['mirrored'] = True
            rotation = rotation[1:]

        # rotation allows float
        rotation_d['degrees'] = float(rotation)

        return rotation_d
