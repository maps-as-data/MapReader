from unittest.mock import patch
import pytest
import requests

from piffle import image


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
    'complex': 'http://imgserver.co/img1/2560,2560,256,/256,/!90/default.jpg',
    'bad_size': '%s/%s/full/a,/0/default.jpg' % (api_endpoint, image_id),
    'bad_region': '%s/%s/200,200/full/0/default.jpg' % (api_endpoint, image_id)
}

sample_image_info = {
    '@context': "http://image.io/api/image/2/context.json",
    '@id': VALID_URLS['simple'],
    'height': 3039,
    'width': 2113,
}


def get_test_imgclient():
    return image.IIIFImageClient(api_endpoint=api_endpoint,
                                 image_id=image_id)


class TestIIIFImageClient:

    def test_defaults(self):
        img = get_test_imgclient()
        # default image url
        assert '%s/%s/full/full/0/default.jpg' % (api_endpoint, image_id) \
               == str(img)
        # info url
        assert '%s/%s/info.json' % (api_endpoint, image_id) \
               == str(img.info())

    def test_outputs(self):
        img = get_test_imgclient()
        # str and unicode should be equivalent
        assert str(img.info()) == str(str(img.info()))
        # repr should have class and image id
        assert 'IIIFImageClient' in repr(img)
        assert img.get_image_id() in repr(img)

    def test_init_opts(self):
        test_opts = {'region': '2560,2560,256,256', 'size': '256,',
                     'rotation': '90', 'quality': 'color', 'fmt': 'png'}
        img = image.IIIFImageClient(api_endpoint=api_endpoint,
                                   image_id=image_id, **test_opts)
        assert img.api_endpoint == api_endpoint
        assert img.image_id == image_id
        expected_img_opts = test_opts.copy()
        del expected_img_opts['region']
        del expected_img_opts['size']
        del expected_img_opts['rotation']
        assert img.image_options == expected_img_opts
        assert str(img.region) == test_opts['region']
        assert str(img.size) == test_opts['size']
        assert str(img.rotation) == test_opts['rotation']

        # TODO: should parse/verify options on init
        # with pytest.raises(image.IIIFImageClientException):
        #     img = image.IIIFImageClient(api_endpoint=api_endpoint, fmt='bogus')

    def test_size(self):
        img = get_test_imgclient()
        width, height, percent = 100, 150, 50
        # width only
        assert '%s/%s/full/%s,/0/default.jpg' % (api_endpoint, image_id, width)\
            == str(img.size(width=width))

        # height only
        assert '%s/%s/full/,%s/0/default.jpg' % \
               (api_endpoint, image_id, height) == \
               str(img.size(height=height))
        # width and height
        assert '%s/%s/full/%s,%s/0/default.jpg' % \
               (api_endpoint, image_id, width, height) == \
               str(img.size(width=width, height=height))
        # exact width and height
        assert '%s/%s/full/!%s,%s/0/default.jpg' % \
               (api_endpoint, image_id, width, height) == \
               str(img.size(width=width, height=height, exact=True))
        # percent
        assert '%s/%s/full/pct:%s/0/default.jpg' % \
               (api_endpoint, image_id, percent) == \
               str(img.size(percent=percent))

    def test_region(self):
        # region options passed through to region object and output
        img = get_test_imgclient()
        x, y, width, height = 5, 10, 100, 150
        assert '%s/%s/%s,%s,%s,%s/full/0/default.jpg' % \
               (api_endpoint, image_id, x, y, width, height) \
            == str(img.region(x=x, y=y, width=width, height=height))

    def test_rotation(self):
        # rotation options passed through to region object and output
        img = get_test_imgclient()
        assert '%s/%s/full/full/90/default.jpg' % \
               (api_endpoint, image_id) \
            == str(img.rotation(degrees=90))
        with pytest.raises(image.IIIFImageClientException):
            img.rotation(foo='bar')

    def test_format(self):
        img = get_test_imgclient()
        png = img.format('png')
        jpg = img.format('jpg')
        gif = img.format('gif')
        assert str(png).endswith('.png')
        assert str(jpg).endswith('.jpg')
        assert str(gif).endswith('.gif')

        with pytest.raises(image.IIIFImageClientException):
            img.format('bogus')

    def test_combine_options(self):
        img = get_test_imgclient()
        width = 100
        fmt = 'png'
        assert '%s/%s/full/%s,/0/default.%s' % \
               (api_endpoint, image_id, width, fmt)\
            == str(img.size(width=width).format(fmt))

        img = get_test_imgclient()
        x, y, width, height = 5, 10, 100, 150
        assert '%s/%s/%s,%s,%s,%s/full/0/default.%s' % \
               (api_endpoint, image_id, x, y, width, height, fmt) \
            == str(img.region(x=x, y=y,
                                  width=width, height=height).format(fmt))

        assert '%s/%s/%s,%s,%s,%s/%s,/0/default.%s' % \
               (api_endpoint, image_id, x, y, width, height, width, fmt) \
            == str(img.size(width=width)
                          .region(x=x, y=y, width=width, height=height)
                          .format(fmt))
        rotation = 90
        assert '%s/%s/%s,%s,%s,%s/%s,/%s/default.%s' % \
               (api_endpoint, image_id, x, y, width, height, width, rotation, fmt) \
            == str(img.size(width=width)
                          .region(x=x, y=y, width=width, height=height)
                          .rotation(degrees=90)
                          .format(fmt))

        # original image object should be unchanged, and still show defaults
        assert '%s/%s/full/full/0/default.jpg' % (api_endpoint, image_id) \
               == str(img)

    def test_init_from_url(self):
        # well-formed
        # - info url
        img = image.IIIFImageClient.init_from_url(VALID_URLS['info'])
        assert isinstance(img, image.IIIFImageClient)
        assert img.image_id == image_id
        # round trip back to original url
        assert str(img.info()) == VALID_URLS['info']
        assert img.api_endpoint == api_endpoint
        # -info with more complex endpoint base url
        img = image.IIIFImageClient.init_from_url(VALID_URLS['info-loris'])
        assert img.api_endpoint == '%s/loris' % api_endpoint
        # - image
        img = image.IIIFImageClient.init_from_url(VALID_URLS['simple'])
        assert isinstance(img, image.IIIFImageClient)
        assert str(img) == VALID_URLS['simple']
        img = image.IIIFImageClient.init_from_url(VALID_URLS['complex'])
        assert str(img) == VALID_URLS['complex']
        assert isinstance(img, image.IIIFImageClient)
        img = image.IIIFImageClient.init_from_url(VALID_URLS['exact'])
        assert str(img) == VALID_URLS['exact']
        assert isinstance(img, image.IIIFImageClient)
        assert img.size.options['exact'] is True

        # malformed
        with pytest.raises(image.ParseError):
            img = image.IIIFImageClient.init_from_url(INVALID_URLS['info'])
        with pytest.raises(image.ParseError):
            image.IIIFImageClient.init_from_url(INVALID_URLS['simple'])
        with pytest.raises(image.ParseError):
            image.IIIFImageClient.init_from_url(INVALID_URLS['complex'])
        with pytest.raises(image.ParseError):
            image.IIIFImageClient.init_from_url(INVALID_URLS['bad_size'])
        with pytest.raises(image.ParseError):
            image.IIIFImageClient.init_from_url(INVALID_URLS['bad_region'])
        with pytest.raises(image.ParseError):
            image.IIIFImageClient.init_from_url('http://info.json')

    def test_as_dicts(self):
        img = image.IIIFImageClient.init_from_url(VALID_URLS['complex'])
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

    @patch('piffle.image.requests')
    def test_image_info(self, mockrequests):
        # test image info logic by mocking requests
        mockrequests.codes.ok = requests.codes.ok
        mockresponse = mockrequests.get.return_value
        mockresponse.status_code = requests.codes.ok
        mockresponse.json.return_value = sample_image_info

        # valid response
        img = image.IIIFImageClient.init_from_url(VALID_URLS['simple'])
        assert img.image_info == sample_image_info
        mockrequests.get.assert_called_with(img.info())
        mockresponse.json.assert_called_with()

        # error response
        mockresponse.status_code = 400
        img = image.IIIFImageClient.init_from_url(VALID_URLS['simple'])
        img.image_info
        mockresponse.raise_for_status.assert_called_with()

    def test_image_width_height(self):
        img = image.IIIFImageClient.init_from_url(VALID_URLS['simple'])

        with patch.object(image.IIIFImageClient, 'image_info',
                          new=sample_image_info):
            assert img.image_width == sample_image_info['width']
            assert img.image_height == sample_image_info['height']

    def test_canonicalize(self):
        img = image.IIIFImageClient.init_from_url(VALID_URLS['simple'])
        square_img_info = sample_image_info.copy()
        square_img_info.update({'height': 100, 'width': 100})
        with patch.object(image.IIIFImageClient, 'image_info',
                          new=square_img_info):
            # square region for square image = full
            img.region.parse('square')
            # percentage: convert to w,h (= 25,25)
            img.size.parse('pct:25')
            img.rotation.parse('90.0')
            assert str(img.canonicalize()) == \
                '%s/%s/full/25,25/90/default.jpg' % (api_endpoint, image_id)


class TestImageRegion:

    def test_defaults(self):
        region = image.ImageRegion()
        assert str(region) == 'full'
        assert region.as_dict() == image.ImageRegion.region_defaults

    def test_init(self):
        # full
        region = image.ImageRegion(full=True)
        assert region.as_dict()['full'] is True
        # square
        region = image.ImageRegion(square=True)
        assert region.as_dict()['square'] is True
        assert region.as_dict()['full'] is False
        # region
        region = image.ImageRegion(x=5, y=7, width=100, height=103)
        assert region.as_dict()['full'] is False
        assert region.as_dict()['x'] == 5
        assert region.as_dict()['y'] == 7
        assert region.as_dict()['width'] == 100
        assert region.as_dict()['height'] == 103
        assert region.as_dict()['percent'] is False
        # percentage region
        region = image.ImageRegion(x=5, y=7, width=100, height=103,
                                   percent=True)
        assert region.as_dict()['full'] is False
        assert region.as_dict()['x'] == 5
        assert region.as_dict()['y'] == 7
        assert region.as_dict()['width'] == 100
        assert region.as_dict()['height'] == 103
        assert region.as_dict()['percent'] is True

        # errors
        with pytest.raises(image.IIIFImageClientException):
            # invalid parameter
            image.ImageRegion(bogus='foo')

            # incomplete options
            image.ImageRegion(x=1)
            image.ImageRegion(x=1, y=2)
            image.ImageRegion(x=1, y=2, w=20)
            image.ImageRegion(percent=True)
            image.ImageRegion().set_options(percent=True, x=1)

            # TODO: type checking? (not yet implemented)

        with pytest.raises(image.ParseError):
            image.ImageRegion().parse('1,2')

    def test_render(self):
        region = image.ImageRegion(full=True)
        assert str(region) == 'full'
        region = image.ImageRegion(square=True)
        assert str(region) == 'square'
        region = image.ImageRegion(x=5, y=5, width=100, height=100)
        assert str(region) == '5,5,100,100'
        region = image.ImageRegion(x=5, y=5, width=100, height=100,
                                   percent=True)
        assert str(region) == 'pct:5,5,100,100'
        region = image.ImageRegion(x=5.1, y=3.14, width=100.76, height=100.89,
                                   percent=True)
        assert str(region) == 'pct:5.1,3.14,100.76,100.89'

    def test_parse(self):
        region = image.ImageRegion()
        # full
        region_str = 'full'
        region.parse(region_str)
        assert str(region) == region_str  # round trip
        assert region.as_dict()['full'] is True
        # square
        region_str = 'square'
        region.parse(region_str)
        assert str(region) == region_str  # round trip
        assert region.as_dict()['full'] is False
        assert region.as_dict()['square'] is True
        # region
        x, y, w, h = [5, 7, 100, 200]
        region_str = '%d,%d,%d,%d' % (x, y, w, h)
        region.parse(region_str)
        assert str(region) == region_str  # round trip
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
        assert str(region) == region_str  # round trip
        region_opts = region.as_dict()
        assert region_opts['full'] is False
        assert region_opts['square'] is False
        assert region_opts['x'] == x
        assert region_opts['y'] == y
        assert region_opts['width'] == w
        assert region_opts['height'] == h
        assert region_opts['percent'] is True

        # invalid or incomplete region strings
        with pytest.raises(image.ParseError):
            region.parse('pct:1,3,')
        with pytest.raises(image.ParseError):
            region.parse('one,two,three,four')

    def test_canonicalize(self):
        # any canonicalization that requires image dimensions to calculate
        # should raise an error
        region = image.ImageRegion()
        region.parse('square')
        with pytest.raises(image.IIIFImageClientException):
            region.canonicalize()

        img = image.IIIFImageClient.init_from_url(VALID_URLS['simple'])
        # full to full - trivial canonicalization
        img.region.canonicalize()
        assert str(img.region) == 'full'
        # x,y,w,h should be preserved as is
        dimensions = '0,0,200,250'
        img.region.parse(dimensions)
        img.region.canonicalize()
        # round trip, should be the same
        assert str(img.region) == dimensions

        # test with square image size
        square_img_info = sample_image_info.copy()
        square_img_info.update({'height': 100, 'width': 100})
        with patch.object(image.IIIFImageClient, 'image_info',
                          new=square_img_info):
            # square requested, image is square = full
            img.region.parse('square')
            img.region.canonicalize()
            assert str(img.region) == 'full'

            # percentages
            img.region.parse('pct:10,1,50,75')
            img.region.canonicalize()
            assert str(img.region) == '10,1,50,75'

            # percentages should be converted to integers
            img.region.parse('pct:10,1,50.5,75.3')
            assert str(img.region) == 'pct:10,1,50.5,75.3'
            img.region.canonicalize()
            assert str(img.region) == '10,1,50,75'

        # test with square with non-square image size
        tall_img_info = sample_image_info.copy()
        tall_img_info.update({'width': 100, 'height': 150})
        with patch.object(image.IIIFImageClient, 'image_info',
                          new=tall_img_info):
            # square requested, should convert to x,y,w,h
            img.region.parse('square')
            img.region.canonicalize()
            assert str(img.region) == '0,25,100,100'

        wide_img_info = sample_image_info.copy()
        wide_img_info.update({'width': 200, 'height': 50})
        with patch.object(image.IIIFImageClient, 'image_info',
                          new=wide_img_info):
            # square requested, should convert to x,y,w,h
            img.region.parse('square')
            img.region.canonicalize()

            assert str(img.region) == '75,0,50,50'


class TestImageSize:

    def test_defaults(self):
        size = image.ImageSize()
        assert str(size) == 'full'
        assert size.as_dict() == image.ImageSize.size_defaults

    def test_init(self):
        # full
        size = image.ImageSize(full=True)
        assert size.as_dict()['full'] is True
        # max
        size = image.ImageSize(max=True)
        assert size.as_dict()['max'] is True
        assert size.as_dict()['full'] is False
        # percentage
        size = image.ImageSize(percent=50)
        assert size.as_dict()['full'] is False
        assert size.as_dict()['percent'] == 50
        # width only
        size = image.ImageSize(width=100)
        assert size.as_dict()['width'] == 100
        # height only
        size = image.ImageSize(height=200)
        assert size.as_dict()['height'] == 200

        # errors
        with pytest.raises(image.IIIFImageClientException):
            # invalid parameter
            image.ImageSize(bogus='foo')

            # incomplete options ?
            # type checking? (not yet implemented)

    def test_render(self):
        size = image.ImageSize(full=True)
        assert str(size) == 'full'
        size = image.ImageSize(max=True)
        assert str(size) == 'max'
        size = image.ImageSize(percent=50)
        assert str(size) == 'pct:50'
        size = image.ImageSize(width=100, height=105)
        assert str(size) == '100,105'
        size = image.ImageSize(width=100)
        assert str(size) == '100,'
        size = image.ImageSize(height=105)
        assert str(size) == ',105'

    def test_parse(self):
        size = image.ImageSize()
        # full
        size_str = 'full'
        size.parse(size_str)
        assert str(size) == size_str  # round trip
        assert size.as_dict()['full'] is True
        # max
        size_str = 'max'
        size.parse(size_str)
        assert str(size) == size_str  # round trip
        assert size.as_dict()['full'] is False
        assert size.as_dict()['max'] is True
        # width and height
        w, h = [100, 200]
        size_str = '%d,%d' % (w, h)
        size.parse(size_str)
        assert str(size) == size_str  # round trip
        size_opts = size.as_dict()
        assert size_opts['full'] is False
        assert size_opts['max'] is False
        assert size_opts['width'] == w
        assert size_opts['height'] == h
        # percentage size
        size_str = 'pct:55'
        size.parse(size_str)
        assert str(size) == size_str  # round trip
        size_opts = size.as_dict()
        assert size_opts['full'] is False
        assert size_opts['percent'] == 55

        # invalid or incomplete size strings
        with pytest.raises(image.ParseError):
            size.parse('pct:')
        with pytest.raises(image.ParseError):
            size.parse('one,two')

    def test_canonicalize(self):
        # any canonicalization that requires image dimensions to calculate
        # should raise an error
        size = image.ImageSize()
        size.parse(',5')
        with pytest.raises(image.IIIFImageClientException):
            size.canonicalize()

        img = image.IIIFImageClient.init_from_url(VALID_URLS['simple'])
        # full to full - trivial canonicalization
        img.size.canonicalize()
        assert str(img.size) == 'full'

        # test sizes with square image size
        square_img_info = sample_image_info.copy()
        square_img_info.update({'height': 100, 'width': 100})
        with patch.object(image.IIIFImageClient, 'image_info',
                          new=square_img_info):
            # requested as ,h - convert to w,
            img.size.parse(',50')
            img.size.canonicalize()
            assert str(img.size) == '50,'

            # percentage: convert to w,h
            img.size.parse('pct:25')
            img.size.canonicalize()
            assert str(img.size) == '25,25'

            # exact
            img.size.parse('!50,50')
            img.size.canonicalize()
            assert str(img.size) == '50,50'

        # test sizes with rectangular image size
        rect_img_info = sample_image_info.copy()
        rect_img_info.update({'width': 50, 'height': 100})
        with patch.object(image.IIIFImageClient, 'image_info',
                          new=rect_img_info):
            img.size.parse('!50,50')
            img.size.canonicalize()
            assert str(img.size) == '25,50'


class TestImageRotation:

    def test_defaults(self):
        rotation = image.ImageRotation()
        assert str(rotation) == '0'
        assert rotation.as_dict() == image.ImageRotation.rotation_defaults

    def test_init(self):
        # degrees
        rotation = image.ImageRotation(degrees=90)
        assert rotation.as_dict()['degrees'] == 90
        assert rotation.as_dict()['mirrored'] is False
        rotation = image.ImageRotation(degrees=95, mirrored=True)
        assert rotation.as_dict()['degrees'] == 95
        assert rotation.as_dict()['mirrored'] is True

    def test_render(self):
        rotation = image.ImageRotation()
        assert str(rotation) == '0'
        rotation = image.ImageRotation(degrees=90)
        assert str(rotation) == '90'
        rotation = image.ImageRotation(degrees=95, mirrored=True)
        assert str(rotation) == '!95'

        # canonicalization
        # - trim any trailing zeros in a decimal value
        assert str(image.ImageRotation(degrees=93.0)) == '93'
        # - leading zero if less than 1
        assert str(image.ImageRotation(degrees=0.05)) == '0.05'
        # - ! if mirrored, followed by integer if possible
        rotation = image.ImageRotation(degrees=95.00, mirrored=True)
        assert str(rotation) == '!95'
        # explicitly test canonicalize method, even though it does nothing
        rotation.canonicalize()
        assert str(rotation) == '!95'

    def test_parse(self):
        rotation = image.ImageRotation()
        rotation_str = '180'
        rotation.parse(rotation_str)
        assert str(rotation) == rotation_str  # round trip
        rotation_str = '!90'
        rotation.parse(rotation_str)
        assert str(rotation) == rotation_str  # round trip
        assert rotation.as_dict()['mirrored'] is True


@pytest.mark.skip
def test_deprecated():
    with pytest.raises(DeprecationWarning) as warn:
        from piffle import iiif
    assert 'piffle.iiif is deprecated' in str(warn)
