from __future__ import annotations

import pathlib
import re
from ast import literal_eval

import geopandas as gpd
import pandas as pd
from shapely import Geometry, from_wkt


def eval_dataframe(df: pd.DataFrame | gpd.GeoDataFrame):
    """Evaluates the columns of a DataFrame/GeoDataFrame and converts them to their respective types.

    Parameters
    ----------
    df : pd.DataFrame | gpd.GeoDataFrame
        The DataFrame/GeoDataFrame to evaluate.

    Returns
    -------
    pd.DataFrame | gpd.GeoDataFrame
        The evaluated DataFrame/GeoDataFrame.
    """
    for col in df.columns:
        try:
            df[col] = df[col].apply(literal_eval)
        except (ValueError, TypeError, SyntaxError):
            pass
    return df


def load_from_csv(
    fpath: str,
    index_col: int | str | None = 0,
    **kwargs,
):
    check_exists(fpath)
    df = pd.read_csv(fpath, index_col=index_col, **kwargs)
    if "polygon" in df.columns:
        print("[INFO] Renaming 'polygon' to 'geometry'.")
        df = df.rename(columns={"polygon": "geometry"})
    df = eval_dataframe(df)
    df = get_geodataframe(df)
    return df


def load_from_geojson(
    fpath: str,
    **kwargs,
):
    check_exists(fpath)
    engine = kwargs.pop("engine", "pyogrio")
    df = gpd.read_file(fpath, engine=engine, **kwargs)
    if "image_id" in df.columns:
        df.set_index("image_id", drop=True, inplace=True)
    elif "name" in df.columns:
        df["image_id"] = df["name"]
        df.set_index("image_id", drop=True, inplace=True)
    df = eval_dataframe(df)
    return df


def load_from_excel(
    fpath: str,
    index_col: int | str | None = 0,
    **kwargs,
):
    check_exists(fpath)
    df = pd.read_excel(fpath, index_col=index_col, **kwargs)
    if "polygon" in df.columns:
        print("[INFO] Renaming 'polygon' to 'geometry'.")
        df = df.rename(columns={"polygon": "geometry"})
    df = eval_dataframe(df)
    df = get_geodataframe(df)
    return df


def get_load_function(
    fpath: str | pathlib.Path,
    **kwargs,
):
    """Find function to load a DataFrame/GeoDataFrame from a file path.

    Parameters
    ----------
    fpath : str or pathlib.Path
        The file path to load the DataFrame/GeoDataFrame from. Can be a CSV/TSV/etc., or Excel or JSON/GeoJSON file.
    """
    check_exists(fpath)

    if re.search(r"\.xls.*$", str(fpath)):  # xls or xlsx
        func = load_from_excel
    elif re.search(r"\..?sv$", str(fpath)):  # csv, tsv, etc
        func = load_from_csv
    elif re.search(r"\..*?json$", str(fpath)):  # json, geojson
        func = load_from_geojson
    else:
        raise ValueError(
            "[ERROR] File format not supported. Please load your file manually."
        )

    return func


def check_exists(fpath):
    if not pathlib.Path(fpath).exists():
        raise FileNotFoundError(f"[ERROR] File {fpath} not found.")


def get_geodataframe(df: pd.DataFrame):
    if "geometry" in df.columns:
        # convert geometry column to shapely objects
        if not isinstance(df.iloc[0]["geometry"], Geometry):
            df["geometry"] = df["geometry"].apply(from_wkt)
        # get crs
        if "crs" in df.columns:
            if df["crs"].nunique() > 1:
                raise ValueError("[ERROR] Multiple CRS found in the dataframe.")
            crs = df["crs"].unique()[0]
        else:
            print("[WARNING] No CRS found for patch images. Setting CRS to EPSG:4326.")
            crs = "EPSG:4326"
        return gpd.GeoDataFrame(df, crs=crs, geometry="geometry")
    return df
