# Change & Version Information

## 0.4

* Dropped support for Python versions 2.7, 3.4, 3.5
* Now tested against python 3.7 and 3.8
* Moved continues integration from Travis-CI to GitHub Actions
* Renamed `piffle.iiif` to `piffle.image`, but for backwards compatibility `piffle.iiif` will still work
* Now includes `piffle.presentation` for simple read access to IIIF Presentation content

## 0.3.2

* Dropped support for Python 3.3

## 0.3

* Now Python 3 compatible
* URI canonicalization for size, region, rotation, and URL as a whole

## 0.2.1

* Bug fix: chaining multiple different options combines all of them properly and does not modify
   the original image object.

## 0.2

* New methods to parse urls and provide image option information. Contributed by [Graham Hukill (@ghukill)](https://github.com/ghukill) [PR #1](https://github.com/emory-lits-labs/piffle/pull/1)
* New method to parse a IIIF Image url and initialize IIIFImageClient via url
* New methods to make IIIF Image options available as dictionary
* Options are now stored internally in logical, dictionary form rather than as IIIF option strings

## 0.1

Initial alpha release, extracting basic IIIF Image API client from [readux codebase](https://github.com/emory-libraries/readux)

* Image client can handle custom id and generates urls for json info, and custom sizes and formats.