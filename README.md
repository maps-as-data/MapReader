# piffle

Python library for generating and parsing [IIIF Image API](http://iiif.io/api/image/2.1/) URLs in an
object-oriented, pythonic fashion.

[![Unit Test Status](https://github.com/Princeton-CDH/piffle/workflows/unit_tests/badge.svg)](https://github.com/Princeton-CDH/piffle/actions?query=workflow%3Aunit_tests)
[![codecov](https://codecov.io/gh/Princeton-CDH/piffle/branch/main/graph/badge.svg)](https://codecov.io/gh/Princeton-CDH/piffle)
[![Maintainability](https://api.codeclimate.com/v1/badges/d37850d90592f9d628df/maintainability)](https://codeclimate.com/github/Princeton-CDH/piffle/maintainability)


Piffle is tested on Python 3.6-3.8.

Piffle was originally developed by Rebecca Sutton Koeser at Emory University as a part of [Readux](https://github.com/ecds/readux>) and forked as a separate project under [emory-lits-labs](https://github.com/emory-lits-labs/). It was later transferred to Rebecca Sutton Koeser at the Center for Digital Humanities at Princeton.

## Installation and example use:

`pip install piffle`

Example use for generating an IIIF image url:

```
>>> from piffle.image import IIIFImageClient
>>> myimg = IIIFImageClient('http://image.server/path/', 'myimgid')
>>> print myimg
http://image.server/path/myimgid/full/full/0/default.jpg
>>> print myimg.info()
http://image.server/path/myimgid/info.json"
>>> print myimg.size(width=120).format('png')
http://image.server/path/myimgid/full/120,/0/default.png
```

Example use for parsing an IIIF image url:

```
>>> from piffle.image import IIIFImageClient
>>> myimg = IIIFImageClient.init_from_url('http://www.example.org/image-service/abcd1234/full/full/0/default.jpg')
>>> print myimg
http://www.example.org/image-service/abcd1234/full/full/0/default.jpg
>>> print myimg.info()
http://www.example.org/image-service/abcd1234/info.json
>>> myimg.as_dict()['size']['full']
True
>>> myimg.as_dict()['size']['exact']
False
>>> myimg.as_dict()['rotation']['degrees']
0.0
```

Example use for reading a IIIF manifest:

```
>>> from piffle.image import IIIFImageClient
>>> from piffle.presentation import IIIFPresentation
>>>  manifest = IIIFPresentation.from_url('https://iiif.bodleian.ox.ac.uk/iiif/manifest/60834383-7146-41ab-bfe1-48ee97bc04be.json')
>>> manifest.label
'Bodleian Library MS. Bodl. 264'
>>> manifest.id
'https://iiif.bodleian.ox.ac.uk/iiif/manifest/60834383-7146-41ab-bfe1-48ee97bc04be.json'
>>> manifest.type
'sc:Manifest'
>>> for canvas in manifest.sequences[0].canvases[:5]:
...     image_id = canvas.images[0].resource.id
...     iiif_img = IIIFImageClient(*image_id.rsplit('/', 1))
...     print(str(iiif_img.size(height=250)))
...
https://iiif.bodleian.ox.ac.uk/iiif/image/90701d49-5e0c-4fb5-9c7d-45af96565468/full/,250/0/default.jpg
https://iiif.bodleian.ox.ac.uk/iiif/image/e878cc78-acd3-43ca-ba6e-90a392f15891/full/,250/0/default.jpg
https://iiif.bodleian.ox.ac.uk/iiif/image/0f1ed064-a972-4215-b884-d8d658acefc5/full/,250/0/default.jpg
https://iiif.bodleian.ox.ac.uk/iiif/image/6fe52b9a-5bb7-4b5b-bbcd-ad0489fcad2a/full/,250/0/default.jpg
https://iiif.bodleian.ox.ac.uk/iiif/image/483ff8ec-347d-4070-8442-dbc15bc7b4de/full/,250/0/default.jpg
```

## Development and Testing

This project uses [git-flow](https://github.com/nvie/gitflow) branching conventions.

Install locally for development (the use of virtualenv is recommended):

`pip install -e .`

Install test dependencies:

`pip install -e ".[test]"`

Run unit tests: `py.test` or `python setup.py test`

## Publishing

To upload a tagged release to [PyPI](https://pypi.python.org/pypi) with
a [wheel](http://pythonwheels.com/) package:

  `python setup.py sdist bdist_wheel upload`
