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
    'complex': '%s/%s/2560,2560,256,256/256,/!90/default.jpg' % (api_endpoint, image_id)
}

INVALID_URLS = {
    'info': 'http://img1/info.json',
    'simple': 'http://imgserver.co/img1/foobar/default.jpg',
    'complex': 'http://imgserver.co/img1/2560,2560,256,256/256,/!90/default.jpg'
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

    def test_init_opts(self):
        test_opts = {'region': '2560,2560,256,256', 'size': '256,',
                     'rotation': '90', 'quality': 'color', 'fmt': 'png'}
        img = iiif.IIIFImageClient(api_endpoint=api_endpoint,
                                   image_id=image_id, **test_opts)
        assert img.api_endpoint == api_endpoint
        assert img.image_id == image_id
        assert img.image_options == test_opts

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

        # malformed
        with pytest.raises(iiif.URLParseError):
            img = iiif.IIIFImageClient.init_from_url(INVALID_URLS['info'])
            img = iiif.IIIFImageClient.init_from_url(INVALID_URLS['simple'])
            img = iiif.IIIFImageClient.init_from_url(INVALID_URLS['complex'])

    def test_region_as_dict(self):
        img = iiif.IIIFImageClient.init_from_url(VALID_URLS['complex'])
        assert img.region_as_dict() == {'full': False, 'h': 256, 'pct': False,
                                        'w': 256, 'x': 2560, 'y': 2560}

    def test_size_as_dict(self):
        img = iiif.IIIFImageClient.init_from_url(VALID_URLS['complex'])
        assert img.size_as_dict() == {'exact': False, 'full': False,
                                      'h': None, 'pct': False, 'w': 256}

    def test_rotation_as_dict(self):
        img = iiif.IIIFImageClient.init_from_url(VALID_URLS['complex'])
        assert img.rotation_as_dict() == {'degrees': 90.0, 'mirrored': True}

    def test_dict_opts(self):
        img = iiif.IIIFImageClient.init_from_url(VALID_URLS['complex'])
        assert img.dict_opts() == {
            'region': {
                'full': False,
                'h': 256,
                'pct': False,
                'w': 256,
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
                'h': None,
                'pct': False,
                'w': 256
            }
        }
