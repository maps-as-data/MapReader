# Change & Version Information

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