"""
A quick prototype to demonstrate the translation of data into the spec being
devised for the ARDC-GMRT project.
"""

from datetime import datetime, timezone
from pathlib import Path
import zipfile
import tempfile
from typing import Any, Dict, Tuple, Union, List
import json
import uuid

import click
import numpy
import pandas
import tiledb
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
import uritools

import extract_gsf
from extract_gsf import RecordTypes


_LOG = structlog.get_logger()


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


def process_gsf(
    pathname,
) -> Union[Tuple[Dict[str, List[datetime]], Dict[str, Any]], Tuple[None, None]]:
    """
    Small util for processing a GSF file to retrieve the ping data and some basic metadata.
    """
    _LOG.info("opening gsf", pathname=pathname)
    with open(pathname, "rb") as stream:
        file_info = extract_gsf.file_info(stream)

        # ideally we should check to see if we have ping records
        find = [
            r for r in file_info if r.record_type == RecordTypes.GSF_SWATH_BATHYMETRY_PING
        ]
        if len(find) == 0:
            _LOG.info("no ping records found; skipping")
            return None, None

        pings = extract_gsf.SwathBathymetryPing(find[0])

        _LOG.info("reading ping records")
        # TODO; return ping header if it's really needed
        dataframe = pings.read_records(stream)

        # any other data to return???
        metadata_records = [
            RecordTypes.GSF_SWATH_BATHY_SUMMARY,
            RecordTypes.GSF_PROCESSING_PARAMETERS,
        ]
        find = [r for r in file_info if r.record_type in metadata_records]

        metadata = {}
        for rtype in find:
            data = rtype.record(0).read(stream)  # should only be one record
            for key, value in data.items():
                metadata[key] = value

        if [
            r
            for r in file_info
            if r.record_type == RecordTypes.GSF_SOUND_VELOCITY_PROFILE
        ]:
            metadata["svp_available"] = True
        else:
            metadata["svp_available"] = False

    return metadata, dataframe


def create_schema(array_uri, dataframe, tiledb_config_pathname) -> Dict[str, Any]:
    """
    Establish the schema first, write later.
    """
    kwargs: Dict[str, Any] = {}
    kwargs["allows_duplicates"] = True

    # dimension and attribute compresison filters
    dim_filters = {}
    attr_filters = {}

    # no real basis on the choice on compression; zstandard is a good alrounder
    for name in dataframe.columns.values:
        attr_filters[name] = tiledb.FilterList([tiledb.ZstdFilter(level=16)])

    # the integer fields i've looked at contain lots of replication
    # so an RLE or delta differencing could work well here
    # other options are bit-width, even byte or bit shuffling should do wonders
    # TODO; compression filter assessment on a variety of data
    for name in dataframe.columns.values:
        if "int" in dataframe[name].dtype.name:
            attr_filters[name] = tiledb.FilterList(
                [tiledb.RleFilter(), tiledb.ZstdFilter(level=16)]
            )

    # define the axis dimensions; depth could also be used here
    # for defining a 3D axis for data indexing and retrieval
    # dim_filters["longitude"] = attr_filters["longitude"]
    # dim_filters["latitude"] = attr_filters["latitude"]
    dim_filters["X"] = attr_filters["X"]
    dim_filters["Y"] = attr_filters["Y"]
    # del attr_filters["longitude"]
    # del attr_filters["latitude"]
    del attr_filters["X"]
    del attr_filters["Y"]

    kwargs["chunksize"] = 20000  # for test, this is multiples of 150 pings
    kwargs["sparse"] = True
    kwargs["dim_filters"] = dim_filters
    kwargs["attr_filters"] = attr_filters
    # kwargs["index_dims"] = ["longitude", "latitude"]
    kwargs["index_dims"] = ["X", "Y"]
    kwargs["mode"] = "schema_only"
    kwargs["full_domain"] = True
    kwargs["cell_order"] = "hilbert"
    kwargs["tile_order"] = "row-major"
    kwargs["capacity"] = 1_000_000

    
    # tiledb config and context; required for S3 access
    ctx = tiledb.Ctx(config=tiledb.Config.load(str(tiledb_config_pathname)))
    kwargs["ctx"] = ctx

    # tiledb have a neat wrapper for pandas dataframes; ___*sweet*___
    tiledb.dataframe_.from_pandas(array_uri, dataframe, **kwargs)

    # change so the data can now be written
    kwargs["mode"] = "append"

    return kwargs


def conversion(
    array_uri: str,
    zip_pathname: Path,
    tiledb_config_pathname: Path,
) -> Dict[str, Any]:
    """
    Small utility to convert the sample GSF files swath ping data into a TileDB point cloud.
    General operations are:
        * Unpack a GSF file from the zip archive
        * Extract the relevant metadata and swath bathymetry ping data
        * Convert to TileDB point cloud
        * Clean up temporary files
    """
    with zipfile.ZipFile(zip_pathname, "r") as src:
        zip_objects = [f for f in src.filelist if Path(f.filename).suffix == ".gsf"]

        # process the first gsf file so we can build the schema
        with tempfile.TemporaryDirectory(suffix=".tmp", prefix="unpack-") as tmpdir:
            src.extract(zip_objects[0], tmpdir)

            gsf_pathname = Path(tmpdir, zip_objects[0].filename)
            general_metadata, dataframe = process_gsf(gsf_pathname)

        # TODO:
        # find out why the bathy summary record has start and end timestamps
        # that are ever so slightly different to the actual start and end
        # pings?
        # the min max bounds in the bathy summary record may also be slightly
        # different to the pings

        # some basic metadata we need are data acquisition datetimes
        # and geographic bounds.
        # calculation of geographic bounds will come later using tiledb or pdal
        datetimes: Dict[str, List[datetime]] = {
            "start_datetime": [general_metadata["time_first_ping"]],
            "end_datetime": [general_metadata["time_last_ping"]],
            "processed_datetime": [general_metadata["processed_datetime"]],
        }

        # unfortunately PDAL doesn't support datetime information
        # drop the column for now. potentially convert to integers at a later date
        # import pdb; pdb.set_trace()
        dataframe.drop("time", inplace=True, axis=1)

        # temporary test
        # cols = ["longitude", "latitude", "depth"]
        # dataframe = dataframe[cols]
        dataframe.rename(columns={"longitude": "X", "latitude": "Y"}, inplace=True)

        # define schema, compression filters, tiling structure etc
        kwargs = create_schema(array_uri, dataframe, tiledb_config_pathname)
        tiledb.dataframe_.from_pandas(array_uri, dataframe, **kwargs)

        # loop over the remainder
        for zip_obj in zip_objects[1:]:
        # for zip_obj in []:

            # extract one gsf into a temp directory and cleanup per iteration
            # keeps disk usage down
            with tempfile.TemporaryDirectory(suffix=".tmp", prefix="unpack-") as tmpdir:
                src.extract(zip_obj, tmpdir)

                gsf_pathname = Path(tmpdir, zip_obj.filename)
                metadata, dataframe = process_gsf(gsf_pathname)

            # append datetime info (calculate temporal bounds later)
            datetimes["start_datetime"].append(metadata["time_first_ping"])
            datetimes["end_datetime"].append(metadata["time_last_ping"])
            datetimes["processed_datetime"].append(metadata["processed_datetime"])

            dataframe.drop("time", inplace=True, axis=1)

            # append to tiledb array
            # dataframe = dataframe[cols]
            dataframe.rename(columns={"longitude": "X", "latitude": "Y"}, inplace=True)
            tiledb.dataframe_.from_pandas(array_uri, dataframe, **kwargs)

    # consolidate the datetimes (i.e. bounds)
    for key, value in time_bounds(datetimes).items():
        general_metadata[key] = value

    # should return some other metadata as well
    return general_metadata


def time_bounds(datetimes: Dict[str, List[datetime]]) -> Dict[str, datetime]:
    """Retrieve the min and max datetimes"""
    dataframe = pandas.DataFrame(datetimes)
    result = {}

    result["start_datetime"] = dataframe["start_datetime"].min().to_pydatetime()
    result["end_datetime"] = dataframe["end_datetime"].max().to_pydatetime()
    result["processed_datetime"] = dataframe["processed_datetime"].max().to_pydatetime()

    return result


def info(data_uri: str, config_pathname: Path, out_pathname: Path) -> Dict[str, Any]:
    """Executes the PDAL info pipeline on the TileDB data file."""
    info_pipeline = [
        {
            "filename": data_uri,
            "type": "readers.tiledb",
            "override_srs": "EPSG:4326",
            "config_file": str(config_pathname),
            # "chunk_size": 120000000,
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
            # "chunk_size": 120000000,
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
            # "chunk_size": 120000000,
            "config_file": str(config_pathname),
        },
        {
            "type": "filters.hexbin",
            "edge_size": 0.000139,  # 0.5 seconds in length
        },
    ]

    pipeline = pdal.Pipeline(json.dumps(hex_pipeline))
    _LOG.info("calculating hexbin", uri=data_uri)
    _ = pipeline.execute()
    metadata = json.loads(pipeline.metadata)

    _write_json(metadata, out_pathname, task="calculate hexbin", pdal_filter="hexbin")

    return metadata


def stac_metadata(
    general_metadata: Dict[str, Any],
    info_metadata: Dict[str, Any],
    stats_metadata: Dict[str, Any],
    hex_metadata: Dict[str, Any],
    data_uri: uritools.split.SplitResultString,
    out_pathname: Path,
) -> None:
    """Generate STAC metadata."""
    general_md = general_metadata.copy()
    start_datetime = general_md.pop("start_datetime")
    end_datetime = general_md.pop("end_datetime")

    sonar_properties = {f"sonar:{key}": value for key, value in general_md.items()}

    created = datetime.now(timezone.utc)

    bounding_box = [
        info_metadata["metadata"]["filters.info"]["bbox"]["minx"],
        info_metadata["metadata"]["filters.info"]["bbox"]["miny"],
        info_metadata["metadata"]["filters.info"]["bbox"]["minz"],
        info_metadata["metadata"]["filters.info"]["bbox"]["maxx"],
        info_metadata["metadata"]["filters.info"]["bbox"]["maxy"],
        info_metadata["metadata"]["filters.info"]["bbox"]["maxz"],
    ]

    geometry = wkt.loads(hex_metadata["metadata"]["filters.hexbin"]["boundary"])

    item = pystac.Item(
        id=str(uuid.uuid4()),
        datetime=end_datetime,
        geometry=mapping(geometry),
        bbox=bounding_box,
        properties=sonar_properties,
    )

    # common metadata
    item.common_metadata.title = "HMAS_Canberra"
    item.common_metadata.description = (
        "Survey conducted by Deakin University and data converted to a point cloud"
    )
    item.common_metadata.start_datetime = start_datetime
    item.common_metadata.end_datetime = end_datetime
    item.common_metadata.created = created
    item.common_metadata.instruments = [
        "em2040",
    ]
    item.common_metadata.providers = [
        pystac.Provider(name="Deakin University", roles=[pystac.ProviderRole.PRODUCER]),
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
        type=PhenomenologyType.SONAR,
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

    # import pdb
    # pdb.set_trace()

    stac_metadata = item.to_dict()
    _write_json(stac_metadata, out_pathname, task="generate STAC metadata")

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
    default="s3://ardc-gmrt-test-data/HMAS-Canberra/HMAS-Canberra.tiledb",
    help="The URI for the output location of the TileDB data file.",
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
def main(uri_name, zip_pathname, tiledb_config_pathname, outdir) -> None:
    """
    Data conversion process (GSF -> TileDB) for the sample HMAS Canberra data.
    STAC metadata generation.
    """
    zip_pathname = Path(zip_pathname)
    tiledb_config_pathname = Path(tiledb_config_pathname)

    _LOG.info("converting GSF data files")
    general_metadata = conversion(uri_name, zip_pathname, tiledb_config_pathname)

    # _write_json(datetimes, "/home/sixy/tmp/deakin-test/datetimes.json", task="datetime")
    # _write_json(general_metadata, "/home/sixy/tmp/deakin-test/general_metadata.json", task="general_metadata")

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
        general_metadata,
        info_metadata,
        stats_metadata,
        hex_metadata,
        data_uri,
        base_pathname.with_suffix(".stac.json"),
    )


if __name__ == "__main__":
    main()
