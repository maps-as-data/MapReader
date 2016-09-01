from piffle import iiif
import pytest

api_endpoint = 'http://imgserver.co'
image_id = 'img1'

# test iiif image urls
VALID_URLS = {
    'info': '%s/%s/info.json' % (api_endpoint, image_id),
    # longer api endpoint path
    'info-loris': '%s/loris/%s/info.json' % (api_endpoint, image_id),
    'simple': '%s/%s/full/full/0/default.jpg' % (api_endpoint, image_id),
    'complex': '%s/%s/2560,2560,256,256/256,/!90/default.jpg' % (api_endpoint, image_id),
    'exact': '%s/%s/full/!256,256/0/default.jpg' % (api_endpoint, image_id)
}

INVALID_URLS = {
    'info': 'http://img1/info.json',
    'simple': 'http://imgserver.co/img1/foobar/default.jpg',
    'complex': 'http://imgserver.co/img1/2560,2560,256,256/256,/!90/default.jpg',
    'bad_size': '%s/%s/full/a,/0/default.jpg' % (api_endpoint, image_id),
    'bad_region': '%s/%s/200,200/full/0/default.jpg' % (api_endpoint, image_id)
}


def get_test_imgclient():
    return iiif.IIIFImageClient(api_endpoint=api_endpoint,
                                image_id=image_id)


class TestIIIFImageClient:

    def test_defaults(self):
        img = get_test_imgclient()
        # default image url
        assert '%s/%s/full/full/0/default.jpg' % (api_endpoint, image_id) \
               == unicode(img)
        # info url
        assert '%s/%s/info.json' % (api_endpoint, image_id) \
               == unicode(img.info())

    def test_outputs(self):
        img = get_test_imgclient()
        # str and unicode should be equivalent
        assert unicode(img.info()) == unicode(str(img.info()))
        # repr should have class and image id
        assert 'IIIFImageClient' in repr(img)
        assert img.get_image_id() in repr(img)

    def test_init_opts(self):
        test_opts = {'region': '2560,2560,256,256', 'size': '256,',
                     'rotation': '90', 'quality': 'color', 'fmt': 'png'}
        img = iiif.IIIFImageClient(api_endpoint=api_endpoint,
                                   image_id=image_id, **test_opts)
        assert img.api_endpoint == api_endpoint
        assert img.image_id == image_id
        expected_img_opts = test_opts.copy()
        del expected_img_opts['region']
        del expected_img_opts['size']
        del expected_img_opts['rotation']
        print img.image_options
        assert img.image_options == expected_img_opts
        assert unicode(img._region) == test_opts['region']
        assert unicode(img._size) == test_opts['size']
        assert unicode(img._rotation) == test_opts['rotation']

        # TODO: should parse/verify options on init
        # with pytest.raises(iiif.IIIFImageClientException):
        #     img = iiif.IIIFImageClient(api_endpoint=api_endpoint, fmt='bogus')

    def test_size(self):
        img = get_test_imgclient()
        width, height, percent = 100, 150, 50
        # width only
        assert '%s/%s/full/%s,/0/default.jpg' % (api_endpoint, image_id, width)\
            == unicode(img.size(width=width))

        # height only
        assert '%s/%s/full/,%s/0/default.jpg' % \
               (api_endpoint, image_id, height) == \
               unicode(img.size(height=height))
        # width and height
        assert '%s/%s/full/%s,%s/0/default.jpg' % \
               (api_endpoint, image_id, width, height) == \
               unicode(img.size(width=width, height=height))
        # exact width and height
        assert '%s/%s/full/!%s,%s/0/default.jpg' % \
               (api_endpoint, image_id, width, height) == \
               unicode(img.size(width=width, height=height, exact=True))
        # percent
        assert '%s/%s/full/pct:%s/0/default.jpg' % \
               (api_endpoint, image_id, percent) == \
               unicode(img.size(percent=percent))

    def test_region(self):
        # region options passed through to region object and output
        img = get_test_imgclient()
        x, y, width, height = 5, 10, 100, 150
        assert '%s/%s/%s,%s,%s,%s/full/0/default.jpg' % \
               (api_endpoint, image_id, x, y, width, height) \
            == unicode(img.region(x=x, y=y, width=width, height=height))

    def test_rotation(self):
        # rotation options passed through to region object and output
        img = get_test_imgclient()
        assert '%s/%s/full/full/90/default.jpg' % \
               (api_endpoint, image_id) \
            == unicode(img.rotation(degrees=90))
        with pytest.raises(iiif.IIIFImageClientException):
            img.rotation(foo='bar')

    def test_format(self):
        img = get_test_imgclient()
        png = img.format('png')
        jpg = img.format('jpg')
        gif = img.format('gif')
        assert unicode(png).endswith('.png')
        assert unicode(jpg).endswith('.jpg')
        assert unicode(gif).endswith('.gif')

        with pytest.raises(iiif.IIIFImageClientException):
            img.format('bogus')

    def test_init_from_url(self):
        # well-formed
        # - info url
        img = iiif.IIIFImageClient.init_from_url(VALID_URLS['info'])
        assert isinstance(img, iiif.IIIFImageClient)
        assert img.image_id == image_id
        # round trip back to original url
        assert unicode(img.info()) == VALID_URLS['info']
        assert img.api_endpoint == api_endpoint
        # -info with more complex endpoint base url
        img = iiif.IIIFImageClient.init_from_url(VALID_URLS['info-loris'])
        assert img.api_endpoint == '%s/loris' % api_endpoint
        # - image
        img = iiif.IIIFImageClient.init_from_url(VALID_URLS['simple'])
        assert isinstance(img, iiif.IIIFImageClient)
        assert unicode(img) == VALID_URLS['simple']
        img = iiif.IIIFImageClient.init_from_url(VALID_URLS['complex'])
        assert unicode(img) == VALID_URLS['complex']
        assert isinstance(img, iiif.IIIFImageClient)
        img = iiif.IIIFImageClient.init_from_url(VALID_URLS['exact'])
        assert unicode(img) == VALID_URLS['exact']
        assert isinstance(img, iiif.IIIFImageClient)
        assert img._size.options['exact'] is True

        # malformed
        with pytest.raises(iiif.ParseError):
            img = iiif.IIIFImageClient.init_from_url(INVALID_URLS['info'])
            img = iiif.IIIFImageClient.init_from_url(INVALID_URLS['simple'])
            img = iiif.IIIFImageClient.init_from_url(INVALID_URLS['complex'])
            img = iiif.IIIFImageClient.init_from_url(INVALID_URLS['bad_size'])
            img = iiif.IIIFImageClient.init_from_url(INVALID_URLS['bad_region'])

    def test_as_dicts(self):
        img = iiif.IIIFImageClient.init_from_url(VALID_URLS['complex'])
        assert img.as_dict() == {
            'region': {
                'full': False,
                'square': False,
                'height': 256,
                'percent': False,
                'width': 256,
                'x': 2560,
                'y': 2560
            },
            'rotation': {
                'degrees': 90.0,
                'mirrored': True
            },
            'size': {
                'exact': False,
                'full': False,
                'max': False,
                'height': None,
                'percent': None,
                'width': 256
            },
            'quality': 'default',
            'format': 'jpg'
        }


class TestImageRegion:

    def test_defaults(self):
        region = iiif.ImageRegion()
        assert unicode(region) == 'full'
        assert region.as_dict() == iiif.ImageRegion.region_defaults

    def test_init(self):
        # full
        region = iiif.ImageRegion(full=True)
        assert region.as_dict()['full'] is True
        # square
        region = iiif.ImageRegion(square=True)
        assert region.as_dict()['square'] is True
        assert region.as_dict()['full'] is False
        # region
        region = iiif.ImageRegion(x=5, y=7, width=100, height=103)
        assert region.as_dict()['full'] is False
        assert region.as_dict()['x'] == 5
        assert region.as_dict()['y'] == 7
        assert region.as_dict()['width'] == 100
        assert region.as_dict()['height'] == 103
        assert region.as_dict()['percent'] is False
        # percentage region
        region = iiif.ImageRegion(x=5, y=7, width=100, height=103,
                                  percent=True)
        assert region.as_dict()['full'] is False
        assert region.as_dict()['x'] == 5
        assert region.as_dict()['y'] == 7
        assert region.as_dict()['width'] == 100
        assert region.as_dict()['height'] == 103
        assert region.as_dict()['percent'] is True

        # errors
        with pytest.raises(iiif.IIIFImageClientException):
            # invalid parameter
            iiif.ImageRegion(bogus='foo')

            # incomplete options
            iiif.ImageRegion(x=1)
            iiif.ImageRegion(x=1, y=2)
            iiif.ImageRegion(x=1, y=2, w=20)
            iiif.ImageRegion(percent=True)

            # TODO: type checking? (not yet implemented)

    def test_render(self):
        region = iiif.ImageRegion(full=True)
        assert unicode(region) == 'full'
        region = iiif.ImageRegion(square=True)
        assert unicode(region) == 'square'
        region = iiif.ImageRegion(x=5, y=5, width=100, height=100)
        assert unicode(region) == '5,5,100,100'
        region = iiif.ImageRegion(x=5, y=5, width=100, height=100,
                                  percent=True)
        assert unicode(region) == 'pct:5,5,100,100'
        region = iiif.ImageRegion(x=5.1, y=3.14, width=100.76, height=100.89,
                                  percent=True)
        assert unicode(region) == 'pct:5.1,3.14,100.76,100.89'

    def test_parse(self):
        region = iiif.ImageRegion()
        # full
        region_str = 'full'
        region.parse(region_str)
        assert unicode(region) == region_str  # round trip
        assert region.as_dict()['full'] is True
        # square
        region_str = 'square'
        region.parse(region_str)
        assert unicode(region) == region_str  # round trip
        assert region.as_dict()['full'] is False
        assert region.as_dict()['square'] is True
        # region
        x, y, w, h = [5, 7, 100, 200]
        region_str = '%d,%d,%d,%d' % (x, y, w, h)
        region.parse(region_str)
        assert unicode(region) == region_str  # round trip
        region_opts = region.as_dict()
        assert region_opts['full'] is False
        assert region_opts['square'] is False
        assert region_opts['x'] == x
        assert region_opts['y'] == y
        assert region_opts['width'] == w
        assert region_opts['height'] == h
        # percentage region
        region_str = 'pct:%d,%d,%d,%d' % (x, y, w, h)
        region.parse(region_str)
        assert unicode(region) == region_str  # round trip
        region_opts = region.as_dict()
        assert region_opts['full'] is False
        assert region_opts['square'] is False
        assert region_opts['x'] == x
        assert region_opts['y'] == y
        assert region_opts['width'] == w
        assert region_opts['height'] == h
        assert region_opts['percent'] is True

        # invalid or incomplete region strings
        with pytest.raises(iiif.ParseError):
            region.parse('pct:1,3,')
            region.parse('one,two,three,four')


class TestImageSize:

    def test_defaults(self):
        size = iiif.ImageSize()
        assert unicode(size) == 'full'
        assert size.as_dict() == iiif.ImageSize.size_defaults

    def test_init(self):
        # full
        size = iiif.ImageSize(full=True)
        assert size.as_dict()['full'] is True
        # max
        size = iiif.ImageSize(max=True)
        assert size.as_dict()['max'] is True
        assert size.as_dict()['full'] is False
        # percentage
        size = iiif.ImageSize(percent=50)
        assert size.as_dict()['full'] is False
        assert size.as_dict()['percent'] == 50
        # width only
        size = iiif.ImageSize(width=100)
        assert size.as_dict()['width'] == 100
        # height only
        size = iiif.ImageSize(height=200)
        assert size.as_dict()['height'] == 200

        # errors
        with pytest.raises(iiif.IIIFImageClientException):
            # invalid parameter
            iiif.ImageSize(bogus='foo')

            # incomplete options ?
            # type checking? (not yet implemented)

    def test_render(self):
        size = iiif.ImageSize(full=True)
        assert unicode(size) == 'full'
        size = iiif.ImageSize(max=True)
        assert unicode(size) == 'max'
        size = iiif.ImageSize(percent=50)
        assert unicode(size) == 'pct:50'
        size = iiif.ImageSize(width=100, height=105)
        assert unicode(size) == '100,105'
        size = iiif.ImageSize(width=100)
        assert unicode(size) == '100,'
        size = iiif.ImageSize(height=105)
        assert unicode(size) == ',105'

    def test_parse(self):
        size = iiif.ImageSize()
        # full
        size_str = 'full'
        size.parse(size_str)
        assert unicode(size) == size_str  # round trip
        assert size.as_dict()['full'] is True
        # max
        size_str = 'max'
        size.parse(size_str)
        assert unicode(size) == size_str  # round trip
        assert size.as_dict()['full'] is False
        assert size.as_dict()['max'] is True
        # width and height
        w, h = [100, 200]
        size_str = '%d,%d' % (w, h)
        size.parse(size_str)
        assert unicode(size) == size_str  # round trip
        size_opts = size.as_dict()
        assert size_opts['full'] is False
        assert size_opts['max'] is False
        assert size_opts['width'] == w
        assert size_opts['height'] == h
        # percentage size
        size_str = 'pct:55'
        size.parse(size_str)
        assert unicode(size) == size_str  # round trip
        size_opts = size.as_dict()
        assert size_opts['full'] is False
        assert size_opts['percent'] == 55

        # invalid or incomplete size strings
        with pytest.raises(iiif.ParseError):
            size.parse('pct:')
            size.parse('one,two')


class TestImageRotation:

    def test_defaults(self):
        rotation = iiif.ImageRotation()
        assert unicode(rotation) == '0'
        assert rotation.as_dict() == iiif.ImageRotation.rotation_defaults

    def test_init(self):
        # degrees
        rotation = iiif.ImageRotation(degrees=90)
        assert rotation.as_dict()['degrees'] == 90
        assert rotation.as_dict()['mirrored'] is False
        rotation = iiif.ImageRotation(degrees=95, mirrored=True)
        assert rotation.as_dict()['degrees'] == 95
        assert rotation.as_dict()['mirrored'] is True

    def test_render(self):
        rotation = iiif.ImageRotation()
        assert unicode(rotation) == '0'
        rotation = iiif.ImageRotation(degrees=90)
        assert unicode(rotation) == '90'
        rotation = iiif.ImageRotation(degrees=95, mirrored=True)
        assert unicode(rotation) == '!95'

    def test_parse(self):
        rotation = iiif.ImageRotation()
        rotation_str = '180'
        rotation.parse(rotation_str)
        assert unicode(rotation) == rotation_str  # round trip
        rotation_str = '!90'
        rotation.parse(rotation_str)
        assert unicode(rotation) == rotation_str  # round trip
        assert rotation.as_dict()['mirrored'] is True
