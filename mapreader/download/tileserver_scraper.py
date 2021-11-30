#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Scraper for tileserver

The main-part/most of these codes are from the following repo:

https://github.com/stamen/the-ultimate-tile-stitcher

(released under MIT license)

Here, we adapted the functions to run them via Python modules
"""

import asyncio
import aiohttp
import json
import os
from random import random
import shapely.geometry
from PIL import Image
from io import BytesIO

from .tileserver_helpers import tile2latlon, latlon2tile, input_class

import nest_asyncio
nest_asyncio.apply()

# global variable
BASE_WAIT = 0.5

# -------
def tile_idxs_in_poly(poly : shapely.geometry.Polygon, zoom : int):
    min_lon, min_lat, max_lon, max_lat = poly.bounds
    (min_x, max_y), (max_x, min_y) = latlon2tile(min_lat, min_lon, zoom), latlon2tile(max_lat, max_lon, zoom)
    for x in range(int(min_x), int(max_x) + 1):
        for y in range(int(min_y) , int(max_y) + 1):
            nw_pt = tile2latlon(x, y, zoom)[::-1] # poly is defined in geojson form
            ne_pt = tile2latlon(x + 1, y, zoom)[::-1] # poly is defined in geojson form
            sw_pt = tile2latlon(x, y + 1, zoom)[::-1] # poly is defined in geojson form
            se_pt = tile2latlon(x + 1, y + 1, zoom)[::-1] # poly is defined in geojson form
            if any(map(lambda pt : shapely.geometry.Point(pt).within(poly),
                (nw_pt, ne_pt, sw_pt, se_pt))):
                yield x, y
            else:
                continue

# -------
async def fetch_and_save(session : aiohttp.ClientSession, url : str, retries : int, filepath : str, **kwargs):
    wait_for = BASE_WAIT
    for retry in range(retries):
        try:
            response = await session.get(url, params=kwargs)
            response.raise_for_status()
            img = await response.read()
            img = Image.open(BytesIO(img))
            img.save(filepath, compress_level=9)

            #with open(filepath, 'wb') as fp:
            #    fp.write(img)
            return True
        except aiohttp.client_exceptions.ClientResponseError:
            #print('err')
            await asyncio.sleep(wait_for)
            wait_for = wait_for * (1.0 * random() + 1.0)
        except asyncio.TimeoutError:
            pass
    return False

# -------
async def runner(opts):
    failed_urls = []

    os.makedirs(opts.out_dir, exist_ok=True)

    semaphore = asyncio.Semaphore(opts.max_connections)

    for feat in opts.poly['features']:
        poly = shapely.geometry.shape(feat['geometry'])

        async with aiohttp.ClientSession() as session:
            tasks = []
            urls = []
            for x, y in tile_idxs_in_poly(poly, opts.zoom):
                url = opts.url.format(z=opts.zoom, x=x, y=y)
                with (await semaphore):
                    filepath = os.path.join(opts.out_dir, '{}_{}_{}.png'.format(opts.zoom, x, y))
                    if os.path.isfile(filepath):
                        continue
                    ret = fetch_and_save(session, url, opts.retries, filepath)
                    urls.append(url)
                    tasks.append(asyncio.ensure_future(ret))

            res : list = await asyncio.gather(*tasks)
            n_failed = res.count(False)

            for i, url in enumerate(urls):
                if res[i] == False:
                    failed_urls.append(url)

    print('Downloaded {}/{}'.format(len(tasks) - n_failed, len(tasks)))
    return failed_urls

# -------
def scraper(poly, zoom, url, out_dir, max_connections=20, retries=10):
    opts = input_class(name="scraper")

    opts.poly = poly
    opts.zoom = zoom
    opts.url = url
    opts.out_dir = out_dir
    opts.max_connections = max_connections
    opts.retries = retries

    with open(opts.poly, 'r') as geojf:
        opts.poly = json.load(geojf)

    loop = asyncio.get_event_loop()
    failed_urls = loop.run_until_complete(runner(opts))
    if len(failed_urls) > 0:
        with open('failed_urls.txt', 'w') as fp:
            fp.writelines((furl + '\n' for furl in failed_urls))