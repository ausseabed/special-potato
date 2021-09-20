"""
A quick prototype to demonstrate the translation of data into the spec being
devised for the ARDC-GMRT project.
"""

from datetime import datetime, timezone
from dateutil import parser
import zipfile
from pathlib import Path
import json
import tempfile
from typing import Any, Dict
import uuid
import click
import pandas
import pdal
import pystac
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.pointcloud import (
    PointcloudExtension,
    SchemaType,
    PhenomenologyType,
    Schema,
    Statistic,
)
from shapely.geometry import box, mapping
from shapely import wkt
import structlog

_LOG = structlog.get_logger()

FIELD_NAMES = "X,Y,Z,TVU"


def conversion(base_pipeline_pathname: Path, zip_pathname: Path, tempdir: tempfile.TemporaryDirectory) -> None:
    """
    Small utility to convert the sample NIDEM ASCII point data into a TileDB point cloud.
    General operations are:
        * Unpack the ASCII from the zip archive
        * Insert the X, Y, Z, TVU field names
        * Insert the TVU value
        * Serialize the new file (temporarily)
        * Convert
        * Clean up temporary files
    """
    with open(base_pipeline_pathname, "r") as src:
        ingestion_pipeline = json.load(src)

    with zipfile.ZipFile(zip_pathname, "r") as src:
        zip_objects = [f for f in src.filelist if Path(f.filename).suffix == ".txt"]
        xls_zobj = [f for f in src.filelist if Path(f.filename).suffix == ".xlsx"][0]

        # open the spreadsheet
        with src.open(xls_zobj) as xls_obj:
            dataframe = pandas.read_excel(xls_obj)

        for zip_obj in zip_objects:
            _LOG.info("processing", filename=zip_obj.filename)
            out_pathname = Path(tempdir, zip_obj.filename)

            ingestion_pipeline[0]["filename"] = str(out_pathname)

            # extract the survey code; expected format is NIDEM_100_25m.txt
            scode = "_".join(out_pathname.name.split("_")[0:2])
            dataset_record = dataframe[dataframe.SURV_CODE == scode]

            with src.open(zip_obj) as ds:
                data = [line.decode().strip() for line in ds.readlines()]

            # append the TVU to each datapoint, as it is now a data level attribute
            tvu = dataset_record.iloc[0].TVU
            data = [f"{line},{tvu}" for line in data]

            # insert field names so that PDAL can read the file
            data.insert(0, FIELD_NAMES)

            if not out_pathname.parent.exists():
                out_pathname.parent.mkdir(parents=True)

            with open(out_pathname, "w") as out_src:
                out_src.writelines([f"{line}\n" for line in data])

            pipeline = pdal.Pipeline(json.dumps(ingestion_pipeline))
            _LOG.info("running pipeline", filename=zip_obj.filename)
            _ = pipeline.execute()

            # update to append mode
            ingestion_pipeline[1]["append"] = True

    _LOG.info("finished conversion")


def info(data_uri: str, config_pathname: Path, out_pathname: Path) -> Dict[str, Any]:
    """Executes the PDAL info pipeline on the TileDB data file."""
    info_pipeline = [
        {
            "filename": data_uri,
            "type": "readers.tiledb",
            "override_srs": "EPSG:4326",
            "config_file": str(config_pathname),
        },
        {"type": "filters.info"},
    ]

    pipeline = pdal.Pipeline(json.dumps(info_pipeline))
    _LOG.info("accessing metadata", uri=data_uri)
    _ = pipeline.execute()
    metadata = json.loads(pipeline.metadata)

    _LOG.info("writing results from info pipeline", pathname=out_pathname)
    with open(out_pathname, "w") as src:
        json.dump(metadata, src, indent=4)

    return metadata


def stats(data_uri: str, config_pathname: Path, out_pathname: Path) -> Dict[str, Any]:
    """Executes the PDAL info pipeline on the TileDB data file."""
    info_pipeline = [
        {
            "filename": data_uri,
            "type": "readers.tiledb",
            "override_srs": "EPSG:4326",
            "config_file": str(config_pathname),
        },
        {"type": "filters.stats"},
    ]

    pipeline = pdal.Pipeline(json.dumps(info_pipeline))
    _LOG.info("calculating statistics", uri=data_uri)
    _ = pipeline.execute()
    metadata = json.loads(pipeline.metadata)

    _LOG.info("writing results from stats pipeline", pathname=out_pathname)
    with open(out_pathname, "w") as src:
        json.dump(metadata, src, indent=4)

    return metadata
