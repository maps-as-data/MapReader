#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Scraper for tileserver

The main-part/most of these codes are from the following repo:

https://github.com/stamen/the-ultimate-tile-stitcher

(released under MIT license)

Here, we adapted the functions to run them via Python modules
"""

from .tileserver_helpers import tile2latlon, latlon2tile, input_class

import asyncio
import aiohttp
import json
import nest_asyncio
import os
import shapely.geometry

from io import BytesIO
from PIL import Image
from random import random
from typing import Tuple, List, Optional

nest_asyncio.apply()

# global variable
BASE_WAIT = 0.5


def tile_idxs_in_poly(
    poly: shapely.geometry.Polygon, zoom: int
) -> Tuple[int, int]:
    """
    Given a Shapely Polygon object and a zoom level, generate a sequence of
    (x, y) tile indices that intersect with the Polygon.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        The Polygon object to intersect with tile indices.
    zoom : int
        The zoom level of the map.

    Yields
    ------
    Tuple[int, int]
        A tuple of tile indices (x, y) that intersect with the Polygon.

    Raises
    ------
    TypeError
        If the poly parameter is not a Shapely Polygon object.
    """
    min_lon, min_lat, max_lon, max_lat = poly.bounds
    (min_x, max_y), (max_x, min_y) = latlon2tile(
        min_lat, min_lon, zoom
    ), latlon2tile(max_lat, max_lon, zoom)

    for x in range(int(min_x), int(max_x) + 1):
        for y in range(int(min_y), int(max_y) + 1):
            nw_pt = tile2latlon(x, y, zoom)[
                ::-1
            ]  # poly is defined in geojson form
            ne_pt = tile2latlon(x + 1, y, zoom)[
                ::-1
            ]  # poly is defined in geojson form
            sw_pt = tile2latlon(x, y + 1, zoom)[
                ::-1
            ]  # poly is defined in geojson form
            se_pt = tile2latlon(x + 1, y + 1, zoom)[
                ::-1
            ]  # poly is defined in geojson form
            if any(
                map(
                    lambda pt: shapely.geometry.Point(pt).within(poly),
                    (nw_pt, ne_pt, sw_pt, se_pt),
                )
            ):
                yield x, y
            else:
                continue


async def fetch_and_save(
    session: aiohttp.ClientSession,
    url: str,
    retries: int,
    filepath: str,
    **kwargs
) -> bool:
    """
    Fetch an image from the specified URL using the specified aiohttp session,
    and save it to a file. The image is saved to the specified file path. The
    function retries fetching the image for the specified number of times. If
    the image is fetched successfully, the function returns True; otherwise,
    it returns False.

    Parameters
    ----------
    session : aiohttp.ClientSession
        The aiohttp session used to fetch the image.
    url : str
        The URL of the image to fetch.
    retries : int
        The number of times to retry fetching the image.
    filepath : str
        The file path where the image will be saved.
    **kwargs
        Optional keyword arguments that will be passed to the `session.get()`
        method.

    Returns
    -------
    bool
        True if the image is fetched successfully and saved to the specified
        file path, False otherwise.

    Raises
    ------
    aiohttp.ClientError
        If any aiohttp client error occurs during the image fetching process.
    """
    wait_for = BASE_WAIT
    for _ in range(retries):
        try:
            response = await session.get(url, params=kwargs)
            response.raise_for_status()
            img = await response.read()
            img = Image.open(BytesIO(img))
            img.save(filepath, compress_level=9)
            return True
        except aiohttp.client_exceptions.ClientResponseError:
            # print('err')
            await asyncio.sleep(wait_for)
            wait_for = wait_for * (1.0 * random() + 1.0)
        except asyncio.TimeoutError:
            pass
    return False


async def runner(opts: input_class) -> List[str]:
    """
    Downloads tiles from a specified URL and saves them to disk within a
    specified polygon. Returns a list of URLs that failed to download.

    Parameters
    ----------
    opts : input_class
        The options to use for downloading the tiles, of the input_class type.

    Returns
    -------
    List[str]
        A list of URLs that failed to download.
    """
    failed_urls = []

    os.makedirs(opts.out_dir, exist_ok=True)

    semaphore = asyncio.Semaphore(opts.max_connections)

    for feat in opts.poly["features"]:
        poly = shapely.geometry.shape(feat["geometry"])

        async with aiohttp.ClientSession() as session:
            tasks = []
            urls = []
            for x, y in tile_idxs_in_poly(poly, opts.zoom):
                url = opts.url.format(z=opts.zoom, x=x, y=y)
                with await semaphore:
                    filepath = os.path.join(
                        opts.out_dir, "{}_{}_{}.png".format(opts.zoom, x, y)
                    )
                    if os.path.isfile(filepath):
                        continue
                    ret = fetch_and_save(session, url, opts.retries, filepath)
                    urls.append(url)
                    tasks.append(asyncio.ensure_future(ret))

            res: list = await asyncio.gather(*tasks)
            n_failed = res.count(False)

            for i, url in enumerate(urls):
                if res[i] is False:
                    failed_urls.append(url)

    print("Downloaded {}/{}".format(len(tasks) - n_failed, len(tasks)))
    return failed_urls


def scraper(
    poly: str,
    zoom: int,
    url: str,
    out_dir: str,
    max_connections: Optional[int] = 20,
    retries: Optional[int] = 10,
) -> None:
    """
    Downloads tiles from the specified URL and saves them to disk within the
    specified polygon. If any tiles fail to download, writes a list of the
    failed URLs to a file.

    Parameters
    ----------
    poly : str
        The path to the GeoJSON file defining the polygon.
    zoom : int
        The zoom level at which to download tiles.
    url : str
        The URL pattern for the tiles to download.
    out_dir : str
        The directory in which to save the downloaded tiles.
    max_connections : int, optional
        The maximum number of simultaneous connections to use when downloading, by default 20.
    retries : int, optional
        The maximum number of times to retry a failed download, by default 10.

    Returns
    -------
    None
    """
    opts = input_class(name="scraper")

    opts.poly = poly
    opts.zoom = zoom
    opts.url = url
    opts.out_dir = out_dir
    opts.max_connections = max_connections
    opts.retries = retries

    with open(opts.poly, "r") as geojf:
        opts.poly = json.load(geojf)

    loop = asyncio.get_event_loop()

    failed_urls = loop.run_until_complete(runner(opts))
    if len(failed_urls) > 0:
        with open("failed_urls.txt", "w") as fp:
            fp.writelines((furl + "\n" for furl in failed_urls))
