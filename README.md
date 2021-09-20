# special-potato

# Overview
Initially just prototype scripts to demonstrate the conversion of data to TileDB, and harvesting of metadata to output into the STAC schema for the PL019 project.

## NIDEM Conversion

```bash
python nidem_convert.py --help
Usage: nidem_convert.py [OPTIONS]

  Data conversion process (ASCII -> TileDB) for the sample NIDEM data. STAC
  metadata generation.

Options:
  --uri-name TEXT                The URI for the output location of the TileDB
                                 data file.

  --base-pipeline-pathname PATH  The base input PDAL pipeline template for
                                 data conversion.

  --zip-pathname PATH            The pathname to the zip file containing the
                                 sample NIDEM data.

  --tiledb-config-pathname PATH  The pathname to the TileDB config file.
                                 Required for writing to AWS S3.

  --outdir DIRECTORY             The base output directory for storing local
                                 files.

  --help                         Show this message and exit.
```
