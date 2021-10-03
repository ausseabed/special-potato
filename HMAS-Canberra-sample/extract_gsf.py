"""
A toy script that converts GSF files into TileDB.
This is purely for demonstration, and not intended to be used for
anythin but demonstration. A module specifically for converting
GSF files will be developed at an unknown time in future.
"""
import datetime
from pathlib import Path
from enum import Enum
import math
import numpy
import pdb
import attr
from typing import List, Dict, Union
import pandas


CHECKSUM_BIT = 0x80000000
NANO_SECONDS_SF = 1e-9
MAX_RECORD_ID = 12
MAX_BEAM_SUBRECORD_ID = 30


def _not_implemented(*args):
    raise NotImplemented


def create_datetime(seconds, nano_seconds):
    time = datetime.datetime.fromtimestamp(seconds + NANO_SECONDS_SF * nano_seconds, tz=datetime.timezone.utc)
    return time


def record_padding(stream):
    """GSF requires that all records are multiples of 4 bytes."""
    pad = numpy.fromfile(stream, "B", count=stream.tell() % 4)
    return pad


class WGS84Coefficients(Enum):
    """
    https://en.wikipedia.org/wiki/Geographic_coordinate_system
    Essentially:
        * lat_m_sf = A - B * cos(2 * lat) + C  * cos(4 * lat) - D * cos(6 * lat)
        * lon_m_sf = E * cos(lat) - F * cos(3 * lat) + G * cos(5 * lat)
    """
    A = 111132.92
    B = 559.82
    C = 1.175
    D = 0.0023
    E = 111412.84
    F = 93.5
    G = 0.118


class RecordTypes(Enum):
    """The various record type contained within the GSF file."""

    GSF_HEADER = 1
    GSF_SWATH_BATHYMETRY_PING = 2
    GSF_SOUND_VELOCITY_PROFILE = 3
    GSF_PROCESSING_PARAMETERS = 4
    GSF_SENSOR_PARAMETERS = 5
    GSF_COMMENT = 6
    GSF_HISTORY = 7
    GSF_NAVIGATION_ERROR = 8
    GSF_SWATH_BATHY_SUMMARY = 9
    GSF_SINGLE_BEAM_PING = 10
    GSF_HV_NAVIGATION_ERROR = 11
    GSF_ATTITUDE = 12

    @property
    def func_mapper(self):
        func_map = {
            RecordTypes.GSF_HEADER: read_header,
            RecordTypes.GSF_SWATH_BATHYMETRY_PING: bathymetry_ping,
            RecordTypes.GSF_SOUND_VELOCITY_PROFILE: read_svp,
            RecordTypes.GSF_PROCESSING_PARAMETERS: processing_parameters,
            RecordTypes.GSF_SENSOR_PARAMETERS: _not_implemented,
            RecordTypes.GSF_COMMENT: comment,
            RecordTypes.GSF_HISTORY: _not_implemented,
            RecordTypes.GSF_NAVIGATION_ERROR: _not_implemented,
            RecordTypes.GSF_SWATH_BATHY_SUMMARY: swath_bathymetry_summary,
            RecordTypes.GSF_SINGLE_BEAM_PING: _not_implemented,
            RecordTypes.GSF_HV_NAVIGATION_ERROR: _not_implemented,
            RecordTypes.GSF_ATTITUDE: attitude,
        }
        return func_map.get(self)


class BeamSubRecordTypes(Enum):
    """The Swath Bathymetry Ping subrecord ID's."""

    DEPTH = 1
    ACROSS_TRACK = 2
    ALONG_TRACK = 3
    TRAVEL_TIME = 4
    BEAM_ANGLE = 5
    MEAN_CAL_AMPLITUDE = 6
    MEAN_REL_AMPLITUDE = 7
    ECHO_WIDTH = 8
    QUALITY_FACTOR = 9
    RECEIVE_HEAVE = 10
    DEPTH_ERROR = 11  # obselete
    ACROSS_TRACK_ERROR = 12  # obselete
    ALONG_TRACK_ERROR = 13  # obselete
    NOMINAL_DEPTH = 14
    QUALITY_FLAGS = 15
    BEAM_FLAGS = 16
    SIGNAL_TO_NOISE = 17
    BEAM_ANGLE_FORWARD = 18
    VERTICAL_ERROR = 19
    HORIZONTAL_ERROR = 20
    INTENSITY_SERIES = 21
    SECTOR_NUMBER = 22
    DETECTION_INFO = 23
    INCIDENT_BEAM_ADJ = 24
    SYSTEM_CLEANING = 25
    DOPPLER_CORRECTION = 26
    SONAR_VERT_UNCERNTAINTY = 27
    SONAR_HORZ_UNCERTAINTY = 28
    DETECTION_WINDOW = 29
    MEAN_ABS_COEF = 30

    @property
    def dtype_mapper(self):
        dtype_map = {
            BeamSubRecordTypes.DEPTH: ">u",
            BeamSubRecordTypes.ACROSS_TRACK: ">i",
            BeamSubRecordTypes.ALONG_TRACK: ">i",
            BeamSubRecordTypes.TRAVEL_TIME: ">u",
            BeamSubRecordTypes.BEAM_ANGLE: ">i",
            BeamSubRecordTypes.MEAN_CAL_AMPLITUDE: ">i",
            BeamSubRecordTypes.MEAN_REL_AMPLITUDE: ">i",
            BeamSubRecordTypes.ECHO_WIDTH: ">u",
            BeamSubRecordTypes.QUALITY_FACTOR: ">u",
            BeamSubRecordTypes.RECEIVE_HEAVE: ">i",
            BeamSubRecordTypes.DEPTH_ERROR: ">u",
            BeamSubRecordTypes.ACROSS_TRACK_ERROR: ">u",
            BeamSubRecordTypes.ALONG_TRACK_ERROR: ">u",
            BeamSubRecordTypes.NOMINAL_DEPTH: ">u",
            BeamSubRecordTypes.QUALITY_FLAGS: ">u",
            BeamSubRecordTypes.BEAM_FLAGS: ">u",
            BeamSubRecordTypes.SIGNAL_TO_NOISE: ">i",
            BeamSubRecordTypes.BEAM_ANGLE_FORWARD: ">u",
            BeamSubRecordTypes.VERTICAL_ERROR: ">u",
            BeamSubRecordTypes.HORIZONTAL_ERROR: ">u",
            BeamSubRecordTypes.INTENSITY_SERIES: ">i",  # not a single type
            BeamSubRecordTypes.SECTOR_NUMBER: ">i",
            BeamSubRecordTypes.DETECTION_INFO: ">i",
            BeamSubRecordTypes.INCIDENT_BEAM_ADJ: ">i",
            BeamSubRecordTypes.SYSTEM_CLEANING: ">i",
            BeamSubRecordTypes.DOPPLER_CORRECTION: ">i",
            BeamSubRecordTypes.SONAR_VERT_UNCERNTAINTY: ">i",  # dtype not defined in 3.09 pdf
            BeamSubRecordTypes.SONAR_HORZ_UNCERTAINTY: ">i",  # dtype and record not defined in 3.09 pdf
            BeamSubRecordTypes.DETECTION_WINDOW: ">i",  # dtype and record not defined in 3.09 pdf
            BeamSubRecordTypes.MEAN_ABS_COEF: ">i",  # dtype and record not defined in 3.09 pdf
        }
        return dtype_map.get(self)


class SensorSpecific(Enum):
    EM2040 = 149


@attr.s()
class Record:

    record_type: RecordTypes = attr.ib()
    data_size: int = attr.ib()
    checksum_flag: bool = attr.ib()
    index: int = attr.ib()

    def read(self, stream, *args):
        stream.seek(self.index)
        data = self.record_type.func_mapper(stream, self.data_size, self.checksum_flag, *args)
        return data


@attr.s()
class FileRecordIndex:

    record_type: RecordTypes = attr.ib()
    record_count: int = attr.ib(init=False)
    data_size: List[int] = attr.ib(repr=False)
    checksum_flag: List[bool] = attr.ib(repr=False)
    indices: List[int] = attr.ib(repr=False)

    def __attrs_post_init__(self):
        self.record_count = len(self.indices)

    def record(self, index):
        result = Record(
            record_type=self.record_type,
            data_size=self.data_size[index],
            checksum_flag=self.checksum_flag[index],
            index=self.indices[index]
        )
        return result


# a class for reading all records and having a global awareness of previously read records
# eg the scale factors for the swath beam data
# one of the attribs would be a FileRecordIndex
# do we maintain the functional compnent? maybe an extra arg=None for records
# that rely on a previous record being read?
@attr.s()
class SwathBathymetryPing:

    file_record: FileRecordIndex = attr.ib()
    # scale_factors: Union[Dict[str, Any], None] = attr.ib(default=None)
    ping_dataframe: Union[pandas.DataFrame, None] = attr.ib(default=None)
    sensor_dataframe: Union[pandas.DataFrame, None] = attr.ib(default=None)

    def read_records(self, stream) -> pandas.DataFrame:
        # loop over each record
        # get header, dataframe and scale factors
        # if this is the first dataframe, this forms the begining on which to append into
        # store the scale factors and pass it through for the next iteration just in case
        # header, we could insert into the dataframe, or do it in the read record
        rec = self.file_record.record(0)
        ping_header, scale_factors, df = rec.read(stream)
        self.ping_dataframe = df
        for i in range(1, self.file_record.record_count):
            rec = self.file_record.record(i)
            ping_header, scale_factors, df = rec.read(stream, scale_factors)
            # append dataframes ...
            self.ping_dataframe = self.ping_dataframe.append(df, ignore_index=True)

        self.ping_dataframe.reset_index(drop=True, inplace=True)

        # probably don't need to return but ehh
        return self.ping_dataframe


@attr.s()
class PingHeader:

    time: datetime.datetime = attr.ib()
    longitude: float = attr.ib()
    latitude: float = attr.ib()
    num_beams: int = attr.ib()
    center_beam: int = attr.ib()
    ping_flags: int = attr.ib()
    reserved: int = attr.ib()
    tide_corrector: int = attr.ib()
    depth_corrector: int = attr.ib()
    heading: float = attr.ib()
    pitch: float = attr.ib()
    roll: float = attr.ib()
    heave: int = attr.ib()
    course: float = attr.ib()
    speed: float = attr.ib()
    height: int = attr.ib()
    separation: int = attr.ib()
    gps_tide_corrector: int = attr.ib()


def file_info(stream):
    fname = Path(stream.name)
    fsize = fname.stat().st_size
    current_pos = stream.tell()
    stream.seek(0)

    results = {rtype: {"indices": [], "data_size": [], "checksum_flag": []} for rtype in RecordTypes}

    while stream.tell() < fsize:
        data_size, record_id, flag = record_info(stream)
        results[RecordTypes(record_id)]["indices"].append(stream.tell())
        results[RecordTypes(record_id)]["data_size"].append(data_size)
        results[RecordTypes(record_id)]["checksum_flag"].append(flag)

        _ = numpy.fromfile(stream, f"S{data_size}", count=1)
        _ = record_padding(stream)

    stream.seek(current_pos)

    r_index = [
        FileRecordIndex(
            record_type=rtype,
            data_size=results[rtype]["data_size"],
            checksum_flag=results[rtype]["checksum_flag"],
            indices=results[rtype]["indices"]
        )
        for rtype in RecordTypes
    ]

    return r_index


def record_info(stream):
    data_size = numpy.fromfile(stream, ">u4", count=1)[0]
    record_identifier = numpy.fromfile(stream, ">i4", count=1)[0]
    checksum_flag = bool(record_identifier & CHECKSUM_BIT)

    return data_size, record_identifier, checksum_flag


def read_header(stream, data_size, checksum_flag):
    if checksum_flag:
        checksum = numpy.fromfile(stream, ">i4", count=1)[0]

    data = numpy.fromfile(stream, f"S{data_size}", count=1)[0]

    _ = record_padding(stream)

    return data


def _proc_param_parser(value):
    """Convert any strings that have known types such as bools, floats."""
    if isinstance(value, datetime.datetime):  # nothing to do already parsed
        return value

    booleans = {
        "yes": True,
        "no": False,
        "true": True,
        "false": False,
    }

    if "," in value:  # dealing with an array
        array = value.split(",")
        if "." in value:  # assumption on period being a decimal point
            parsed = numpy.array(array, dtype="float").tolist()
        else:
            # should be dealing with an array of "UNKNWN" or "UNKNOWN"
            parsed = ["unknown"]*len(array)
    elif "." in value:  # assumption on period being a decimal point
        parsed = float(value)
    elif value.lower() in booleans:
        parsed = booleans[value.lower()]
    elif value.lower() in ["unknwn", "unknown"]:
        parsed = "unknown"
    else:  # most likely an integer or generic string
        try:
            parsed = int(value)
        except ValueError:
            parsed = value.lower()

    return parsed


def _standardise_proc_param_keys(key):
    """Convert to lowercase, replace any spaces with underscore."""
    return key.lower().replace(" ", "_")


def processing_parameters(stream, data_size, checksum_flag):
    params = dict()
    idx = 0

    blob = stream.readline(data_size)

    if checksum_flag:
        checksum = numpy.frombuffer(blob, ">i4", count=1)[0]
        idx += 4

    dtype = numpy.dtype(
        [
            ("time_seconds", ">i4"),
            ("time_nano_seconds", ">i4"),
            ("num_params", ">i2"),
        ]
    )
    data = numpy.frombuffer(blob, dtype, count=1)
    time_seconds = int(data["time_seconds"][0])
    time_nano_seconds = int(data["time_nano_seconds"][0])
    num_params = data["num_params"][0]

    idx += 10

    for i in range(num_params):
        param_size = numpy.frombuffer(blob[idx:], ">i2", count=1)[0]
        idx += 2
        data = numpy.frombuffer(blob[idx:], f"S{param_size}", count=1)[0]
        idx += param_size

        key, value = data.decode("utf-8").strip().split("=")

        if key == "REFERENCE TIME":
            value = datetime.datetime.strptime(value, "%Y/%j %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
            params["processed_datetime"] = value + datetime.timedelta(
                seconds=time_seconds,
                milliseconds=time_nano_seconds * 1e-6
            )
            
        params[_standardise_proc_param_keys(key)] = _proc_param_parser(value)

    _ = record_padding(stream)

    return params


def attitude(stream, data_size, checksum_flag):
    # using stream.readline() would stop at the first new line
    # using stream.readlines() can read more than data_size
    blob = numpy.fromfile(stream, f"S{data_size}", count=1)[0]
    idx = 0

    if checksum_flag:
        checksum = numpy.frombuffer(blob, ">i4", count=1)[0]
        idx += 4

    base_time = numpy.frombuffer(blob[idx:], ">i4", count=2)
    idx += 8

    acq_time = create_datetime(base_time[0], base_time[1])

    num_measurements = numpy.frombuffer(blob[idx:], ">i2", count=1)[0]
    idx += 2

    data = {
        "attitude_time": [],
        "pitch": [],
        "roll": [],
        "heave": [],
        "heading": [],
    }

    dtype = numpy.dtype(
        [
            ("attitude_time", ">i2"),
            ("pitch", ">i2"),
            ("roll", ">i2"),
            ("heave", ">i2"),
            ("heading", ">u2"),
        ]
    )
    for i in range(num_measurements):
        numpy_blob = numpy.frombuffer(blob[idx:], dtype, count=1)[0]
        idx += 10

        data["attitude_time"].append(
            acq_time + datetime.timedelta(seconds=numpy_blob["attitude_time"] / 1000)
        )
        data["pitch"].append(numpy_blob["pitch"] / 100)
        data["roll"].append(numpy_blob["roll"] / 100)
        data["heave"].append(numpy_blob["heave"] / 100)
        data["heading"].append(numpy_blob["heading"] / 100)

    _ = record_padding(stream)

    return data


def read_svp(stream, data_size, flag):
    dtype = numpy.dtype(
        [
            ("obs_seconds", ">u4"),
            ("obs_nano", ">u4"),
            ("app_seconds", ">u4"),
            ("app_nano", ">u4"),
            ("lon", ">i4"),
            ("lat", ">i4"),
            ("num_points", ">u4"),
        ]
    )

    blob = numpy.fromfile(stream, dtype, count=1)
    svp = numpy.fromfile(stream, ">u4", count=2 * blob["num_points"][0]) / 100

    data = {
        "obs_time": create_datetime(
            blob["obs_seconds"][0], blob["obs_nano"][0]
        ),
        "app_time": create_datetime(
            blob["app_seconds"][0], blob["app_nano"][0]
        ),
        "longitude": blob["lon"][0] / 10_000_000,
        "latitude": blob["lat"][0] / 10_000_000,
        "num_points": blob["num_points"][0],
        "svp_array": svp.reshape((blob["num_points"][0], 2)),
    }

    _ = record_padding(stream)

    return data


def swath_bathymetry_summary(stream, data_size, flag):
    dtype = numpy.dtype(
        [
            ("time_first_ping_seconds", ">i4"),
            ("time_first_ping_nano_seconds", ">i4"),
            ("time_last_ping_seconds", ">i4"),
            ("time_last_ping_nano_seconds", ">i4"),
            ("min_latitude", ">i4"),
            ("min_longitude", ">i4"),
            ("max_latitude", ">i4"),
            ("max_longitude", ">i4"),
            ("min_depth", ">i4"),
            ("max_depth", ">i4"),
        ]
    )

    blob = numpy.fromfile(stream, dtype, count=1)

    data = {
        "time_first_ping": create_datetime(
            blob["time_first_ping_seconds"][0], blob["time_first_ping_nano_seconds"][0]
        ),
        "time_last_ping": create_datetime(
            blob["time_last_ping_seconds"][0], blob["time_last_ping_nano_seconds"][0]
        ),
        "min_latitude": blob["min_latitude"][0] / 10_000_000,
        "min_longitude": blob["min_longitude"][0] / 10_000_000,
        "max_latitude": blob["max_latitude"][0] / 10_000_000,
        "max_longitude": blob["max_longitude"][0] / 10_000_000,
        "min_depth": blob["min_depth"][0] / 100,
        "max_depth": blob["max_depth"][0] / 100,
    }

    _ = record_padding(stream)

    return data


def comment(stream, data_size, flag):
    dtype = numpy.dtype(
        [
            ("time_comment_seconds", ">i4"),
            ("time_comment_nano_seconds", ">i4"),
            ("comment_length", ">i4"),
        ]
    )
    blob = stream.readline(data_size)
    decoded = numpy.frombuffer(blob, dtype, count=1)
    length = decoded["comment_length"][0]

    data = {
        "time": create_datetime(
            decoded["time_comment_seconds"][0], decoded["time_comment_nano_seconds"][0]),
        "length": length,
        "comment": blob[12:].decode().strip().rstrip("\x00"),
    }

    _ = record_padding(stream)

    return data


def _correct_ping_header(data):
    data_dict = {}

    data_dict["time"] = create_datetime(
        data["time_ping_seconds"][0], data["time_ping_nano_seconds"][0]
    )
    data_dict["longitude"] = float(data["longitude"][0] / 10_000_000)
    data_dict["latitude"] = float(data["latitude"][0] / 10_000_000)
    data_dict["num_beams"] = int(data["number_beams"][0])
    data_dict["center_beam"] = int(data["centre_beam"][0])
    data_dict["ping_flags"] = int(data["ping_flags"][0])
    data_dict["reserved"] = int(data["reserved"][0])
    data_dict["tide_corrector"] = int(data["tide_corrector"][0])
    data_dict["depth_corrector"] = int(data["depth_corrector"][0])
    data_dict["heading"] = float(data["heading"][0] / 100)
    data_dict["pitch"] = float(data["pitch"][0] / 100)
    data_dict["roll"] = float(data["roll"][0] / 100)
    data_dict["heave"] = int(data["heave"][0])
    data_dict["course"] = float(data["course"][0] / 100)
    data_dict["speed"] = float(data["speed"][0] / 100)
    data_dict["height"] = int(data["height"][0])
    data_dict["separation"] = int(data["separation"][0])
    data_dict["gps_tide_corrector"] = int(data["gps_tide_corrector"][0])

    ping_header = PingHeader(**data_dict)

    return ping_header


def bathymetry_ping(stream, data_size, flag, scale_factors=None):
    idx = 0
    blob = numpy.fromfile(stream, f"S{data_size}", count=1)[0]

    dtype = numpy.dtype(
        [
            ("time_ping_seconds", ">i4"),
            ("time_ping_nano_seconds", ">i4"),
            ("longitude", ">i4"),
            ("latitude", ">i4"),
            ("number_beams", ">i2"),
            ("centre_beam", ">i2"),
            ("ping_flags", ">i2"),
            ("reserved", ">i2"),
            ("tide_corrector", ">i2"),
            ("depth_corrector", ">i4"),
            ("heading", ">u2"),
            ("pitch", ">i2"),
            ("roll", ">i2"),
            ("heave", ">i2"),
            ("course", ">u2"),
            ("speed", ">u2"),
            ("height", ">i4"),
            ("separation", ">i4"),
            ("gps_tide_corrector", ">i4"),
        ]
    )

    ping_header = _correct_ping_header(numpy.frombuffer(blob, dtype=dtype, count=1))

    idx += 56  # includes 2 bytes of spare space

    # first subrecord
    subrecord_hdr = numpy.frombuffer(blob[idx:], ">i4", count=1)[0]
    subrecord_id = (subrecord_hdr & 0xFF000000) >> 24
    subrecord_size = subrecord_hdr & 0x00FFFFFF

    idx += 4

    if subrecord_id == 100:
        # scale factor subrecord
        num_factors = numpy.frombuffer(blob[idx:], ">i4", count=1)[0]
        idx += 4

        # if we have input sf's return new ones
        # some pings don't store a scale factor record and rely on
        # ones read from a previous ping
        scale_factors = {}  # if we have input sf's return new ones
        for i in range(num_factors):
            sfs_blob = numpy.frombuffer(blob[idx:], ">i4", count=3)
            subid = (sfs_blob[0] & 0xFF000000) >> 24
            compression_flag = (sfs_blob & 0x00FF0000) >> 16

            scale_factors[BeamSubRecordTypes(subid)] = sfs_blob[1:]
            idx += 12

    else:
        if scale_factors is None:
            # can't really do anything sane
            # could return the unscaled data, but that's not the point here
            raise Exception("Record has no scale factors")

        # roll back the index by 4 bytes
        idx -= 4

    subrecord_hdr = numpy.frombuffer(blob[idx:], ">i4", count=1)[0]
    idx += 4

    subrecord_id = (subrecord_hdr & 0xFF000000) >> 24
    subrecord_size = subrecord_hdr & 0x00FFFFFF

    # beam array subrecords
    subrecords = {}
    while subrecord_id <= MAX_BEAM_SUBRECORD_ID:
        size = subrecord_size // ping_header.num_beams
        sub_rec_type = BeamSubRecordTypes(subrecord_id)
        dtype = f"{sub_rec_type.dtype_mapper}{size}"
        sub_rec_blob = numpy.frombuffer(blob[idx:], dtype, count=ping_header.num_beams)

        idx += size * ping_header.num_beams

        data = sub_rec_blob / scale_factors[sub_rec_type][0] - scale_factors[sub_rec_type][1]

        # store as float64 only when really required (we'll arbitrarily use 100_000 as the change)
        # (we'll arbitrarily use 100_000 as the defining limit)
        # most of the data has been truncated or scaled and stored as int's so
        # some level of precision has already been lost
        if scale_factors[sub_rec_type][0] < 100_000:
            data = data.astype("float32")

        subrecords[sub_rec_type] = data

        # info for next subrecord
        subrecord_hdr = numpy.frombuffer(blob[idx:], ">i4", count=1)[0]
        idx += 4

        subrecord_id = (subrecord_hdr & 0xFF000000) >> 24
        subrecord_size = subrecord_hdr & 0x00FFFFFF

    # convert beam arrays to point cloud structure (i.e. generate coords for every beam)
    # see https://math.stackexchange.com/questions/389942/why-is-it-necessary-to-use-sin-or-cos-to-determine-heading-dead-reckoning
    lat_radians = math.radians(ping_header.latitude)
    lat_mtr_sf = WGS84Coefficients.A.value - WGS84Coefficients.B.value * math.cos(2 * lat_radians) + WGS84Coefficients.C.value * math.cos(4 * lat_radians) - WGS84Coefficients.D.value * math.cos(6 * lat_radians)
    lon_mtr_sf = WGS84Coefficients.E.value * math.cos(lat_radians) - WGS84Coefficients.F.value * math.cos(3 * lat_radians) + WGS84Coefficients.G.value * math.cos(5 * lat_radians)

    delta_x = math.sin(math.radians(ping_header.heading))
    delta_y = math.cos(math.radians(ping_header.heading))

    lon2 = ping_header.longitude + delta_y / lon_mtr_sf * subrecords[BeamSubRecordTypes.ACROSS_TRACK] + delta_x / lon_mtr_sf * subrecords[BeamSubRecordTypes.ALONG_TRACK]
    lat2 = ping_header.latitude - delta_x / lat_mtr_sf * subrecords[BeamSubRecordTypes.ACROSS_TRACK] + delta_y / lat_mtr_sf * subrecords[BeamSubRecordTypes.ALONG_TRACK]

    df = pandas.DataFrame({k.name.lower(): v for k, v in subrecords.items()})
    df.insert(0, "latitude", lat2)
    df.insert(0, "longitude", lon2)

    # include the header info in the dataframe as that was desired by many in the survey
    ignore = [
        "longitude",
        "latitude",
        "num_beams",
        "center_beam",
        "reserved",
    ]
    for key, value in attr.asdict(ping_header).items():
        if key in ignore:
            # we don't want to overwrite with a constant for this ping
            continue
        df[key] = value

    # SKIPPING:
    #     * sensor specific sub records
    #     * intensity series
    # print(idx)

    return ping_header, scale_factors, df


def run(fname):
    stream = open(fname, "rb")

    data_size, record_id, flag = record_info(stream)
    header = read_header(stream, data_size, flag)

    fill = numpy.fromfile(stream, "B", count=stream.tell() % 4)

    data_size, record_id, flag = record_info(stream)
    params = processing_parameters(stream, data_size, flag)

    fill = numpy.fromfile(stream, "B", count=stream.tell() % 4)

    # pdb.set_trace()
    att = []
    data_size, record_id, flag = record_info(stream)

    while record_id == 12:
        att.append(attitude(stream, data_size, flag))
        fill = numpy.fromfile(stream, "B", count=stream.tell() % 4)
        data_size, record_id, flag = record_info(stream)

    svp = read_svp(stream, data_size, flag)

    pdb.set_trace()
