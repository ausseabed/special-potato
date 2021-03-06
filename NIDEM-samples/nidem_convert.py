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
import numpy
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
import tiledb
import uritools
import structlog

_LOG = structlog.get_logger()

FIELD_NAMES = "X,Y,Z,TVU"


class Encoder(json.JSONEncoder):
    """Extensible encoder to handle non-json types."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)

        return super(Encoder, self).default(obj)


def _write_json(data: Dict[str, Any], out_pathname: Path, **kwargs):
    """Small util for writing JSON content."""
    _LOG.info("writing JSON file", pathname=str(out_pathname), **kwargs)
    with open(out_pathname, "w") as src:
        json.dump(data, src, indent=4, cls=Encoder)


def conversion(
    data_uri: str,
    base_pipeline_pathname: Path,
    zip_pathname: Path,
    config_pathname: Path,
    tempdir: tempfile.TemporaryDirectory,
) -> None:
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

    ingestion_pipeline[1]["config_file"] = str(config_pathname)
    ingestion_pipeline[1]["filename"] = data_uri

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

    _write_json(metadata, out_pathname, task="calculate info", pdal_pipeline="info")

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

    _write_json(metadata, out_pathname, task="calculate stats", pdal_pipelin="stats")

    return metadata


def hexbin(data_uri: str, config_pathname: Path, out_pathname: Path) -> Dict[str, Any]:
    """
    Get something akin to a convexhull.
    See:
        https://pdal.io/stages/filters.hexbin.html#filters-hexbin
    """
    hex_pipeline = [
        {
            "filename": data_uri,
            "type": "readers.tiledb",
            "override_srs": "EPSG:4326",
            "config_file": str(config_pathname),
        },
        {
            "type": "filters.hexbin",
            "edge_size": 0.020000000015999997,  # 1 minute 12 seconds in length
        },
    ]

    pipeline = pdal.Pipeline(json.dumps(hex_pipeline))
    _LOG.info("calculating hexbin", uri=data_uri)
    _ = pipeline.execute()
    metadata = json.loads(pipeline.metadata)

    _write_json(metadata, out_pathname, task="calculate hexbin", pdal_filter="hexbin")

    return metadata


def stac_metadata(
    info_metadata: Dict[str, Any],
    stats_metadata: Dict[str, Any],
    hex_metadata: Dict[str, Any],
    data_uri: uritools.split.SplitResultString,
    out_pathname: Path,
) -> None:
    """Generate STAC metadata"""
    # see https://cmi.ga.gov.au/data-products/dea/325/dea-intertidal-elevation-landsat
    # for datetime information
    end_datetime = parser.parse("2017-07-31T23:59:59.00Z")
    start_datetime = parser.parse("1986-08-16T00:00:00.00Z")
    created = datetime.now(timezone.utc)

    bounding_box = [
        info_metadata["metadata"]["filters.info"]["bbox"]["minx"],
        info_metadata["metadata"]["filters.info"]["bbox"]["miny"],
        info_metadata["metadata"]["filters.info"]["bbox"]["minz"],
        info_metadata["metadata"]["filters.info"]["bbox"]["maxx"],
        info_metadata["metadata"]["filters.info"]["bbox"]["maxy"],
        info_metadata["metadata"]["filters.info"]["bbox"]["maxz"],
    ]
    # geometry = box(
    #     minx=bounding_box[0],
    #     miny=bounding_box[1],
    #     maxx=bounding_box[3],
    #     maxy=bounding_box[4],
    # )  # incase a bounding box is preferred for spatially referencing datasets
    geometry = wkt.loads(hex_metadata["metadata"]["filters.hexbin"]["boundary"])

    item = pystac.Item(
        id=str(uuid.uuid4()),
        datetime=end_datetime,  # just using the end date as NIDEM is a time composition
        geometry=mapping(geometry),
        bbox=bounding_box,
        properties={},
    )

    # common metadata
    item.common_metadata.title = "NIDEM-25m"
    item.common_metadata.description = "Sourced from NIDEM and converted to a point cloud"
    item.common_metadata.start_datetime = start_datetime
    item.common_metadata.end_datetime = end_datetime
    item.common_metadata.created = created
    item.common_metadata.instruments = [
        "tm",
        "etm+",
        "oli",
        "tirs",
    ]
    item.common_metadata.providers = [
        pystac.Provider(
            name="James Cook University", roles=[pystac.ProviderRole.PRODUCER]
        ),
        pystac.Provider(name="Geoscience Australia", roles=[pystac.ProviderRole.HOST]),
    ]

    proj = ProjectionExtension.ext(item, add_if_missing=True)
    pc = PointcloudExtension.ext(item, add_if_missing=True)

    proj.apply(epsg=4326)

    # schema and stats metadata
    pc_schema = info_metadata["metadata"]["filters.info"]["schema"]["dimensions"]
    pc_stats = stats_metadata["metadata"]["filters.stats"]["statistic"]

    # point cloud extension
    pc.apply(
        count=info_metadata["metadata"]["filters.info"]["num_points"],
        type=PhenomenologyType.EOPC,
        encoding="TileDB",
        schemas=[Schema(properties=sch) for sch in pc_schema],
        statistics=[Statistic(properties=stats) for stats in pc_stats],
    )

    # data and metadata uri population
    stac_uri = uritools.urisplit(
        Path(data_uri.path).parent.joinpath(out_pathname.name).as_uri()
    )
    metadata_target = uritools.uricompose(
        scheme=data_uri.scheme, authority=data_uri.authority, path=stac_uri.path
    )
    item.add_asset("data", pystac.Asset(href=data_uri.geturi(), roles=["data"]))
    item.add_link(
        pystac.Link(rel="self", media_type=pystac.MediaType.JSON, target=metadata_target)
    )

    # ignore validation for this prototype
    # item.validate()

    stac_metadata = item.to_dict()
    _write_json(item.to_dict(), out_pathname, task="generate STAC metadata")

    # attach stac metadata to the tiledb array
    attach_metadata(data_uri.geturi(), "stac", stac_metadata)


def attach_metadata(array_uri: str, label: str, metadata: Dict[str, Any]) -> None:
    """
    Attach the STAC and some other basic metadata to the TileDB array.
    """
    with tiledb.open(array_uri, "w") as ds:
        ds.meta[label] = json.dumps(metadata, indent=4, cls=Encoder)


@click.command()
@click.option(
    "--uri-name",
    default="s3://ardc-gmrt-test-data/NIDEM-25m/NIDEM_25m.tiledb",
    help="The URI for the output location of the TileDB data file.",
)
@click.option(
    "--base-pipeline-pathname",
    type=click.Path(exists=True, readable=True),
    help="The base input PDAL pipeline template for data conversion.",
)
@click.option(
    "--zip-pathname",
    type=click.Path(exists=True, readable=True),
    help="The pathname to the zip file containing the sample NIDEM data.",
)
@click.option(
    "--tiledb-config-pathname",
    type=click.Path(exists=True, readable=True),
    help="The pathname to the TileDB config file. Required for writing to AWS S3.",
)
@click.option(
    "--outdir",
    type=click.Path(file_okay=False, writable=True),
    help="The base output directory for storing local files.",
)
def main(uri_name, base_pipeline_pathname, zip_pathname, tiledb_config_pathname, outdir):
    """
    Data conversion process (ASCII -> TileDB) for the sample NIDEM data.
    STAC metadata generation.
    """
    base_pipeline_pathname = Path(base_pipeline_pathname)
    zip_pathname = Path(zip_pathname)
    tiledb_config_pathname = Path(tiledb_config_pathname)

    _LOG.info("converting ASCII data files")
    with tempfile.TemporaryDirectory(suffix=".tmp", prefix="unpack-") as tempdir:
        conversion(
            uri_name,
            base_pipeline_pathname,
            zip_pathname,
            tiledb_config_pathname,
            tempdir,
        )

    data_uri = uritools.urisplit(uri_name)
    basepath = Path(*(Path(data_uri.path).parts[1:]))  # account for the leading "/"
    base_pathname = Path(outdir, Path(basepath.stem))

    if not base_pathname.parent.exists():
        _LOG.info("creating output directory", pathname=str(base_pathname.parent))
        base_pathname.parent.mkdir(parents=True)

    _LOG.info("executing task `PDAL info pipeline`")
    info_metadata = info(
        uri_name, tiledb_config_pathname, base_pathname.with_suffix(".info.json")
    )

    _LOG.info("executing task `PDAL stats pipeline`")
    stats_metadata = stats(
        uri_name, tiledb_config_pathname, base_pathname.with_suffix(".stats.json")
    )

    _LOG.info("executing task `PDAL hexbin filter`")
    hex_metadata = hexbin(
        uri_name, tiledb_config_pathname, base_pathname.with_suffix(".hexbin.json")
    )

    _LOG.info("executing task `generate stac metadata`")
    stac_metadata(
        info_metadata,
        stats_metadata,
        hex_metadata,
        data_uri,
        base_pathname.with_suffix(".stac.json"),
    )


if __name__ == "__main__":
    main()
