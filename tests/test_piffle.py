from piffle import iiif
import pytest

api_endpoint = 'http://imgserver.co/'
image_id = 'img1'


def get_test_imgclient():
    return iiif.IIIFImageClient(api_endpoint=api_endpoint,
            image_id=image_id)


class TestIIIFImageClient:

    def test_defaults(self):
        img = get_test_imgclient()
        # default image url
        assert '%s%s/full/full/0/default.jpg' % (api_endpoint, image_id) \
                == unicode(img)
        # info url
        assert '%s%s/info.json' % (api_endpoint, image_id) == unicode(img.info())

    def test_size(self):
        img = get_test_imgclient()
        width, height, percent = 100, 150, 50
        # width only
        assert '%s%s/full/%s,/0/default.jpg' % (api_endpoint, image_id, width) \
            == unicode(img.size(width=width))

        # height only
        assert '%s%s/full/,%s/0/default.jpg' % \
               (api_endpoint, image_id, height) == \
               unicode(img.size(height=height))
        # width and height
        assert '%s%s/full/%s,%s/0/default.jpg' % \
               (api_endpoint, image_id, width, height) == \
               unicode(img.size(width=width, height=height))
        # exact width and height
        assert '%s%s/full/!%s,%s/0/default.jpg' % \
               (api_endpoint, image_id, width, height) == \
               unicode(img.size(width=width, height=height, exact=True))
        # percent
        assert '%s%s/full/pct:%s/0/default.jpg' % \
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

        with pytest.raises(Exception):
            img.format('bogus')

    def test_url_parse(self):
        img = iiif.IIIFImageClient()
        # simple API parameters
        img.parse_from_url('http://imgserver.co/loris/img1/full/full/0/default.jpg')
        img.api_params_as_dict()
        # more complex API parameters
        img.parse_from_url('http://imgserver.co/loris/img1/2560,2560,256,256/256,/!90/default.jpg')
        img.api_params_as_dict()


