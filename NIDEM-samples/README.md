# NIDEM

See https://cmi.ga.gov.au/data-products/dea/325/dea-intertidal-elevation-landsat for specifics on NIDEM.
The metadata was harvested from varied sources:
* Supplied Excel Spreasheet
* CMI (see above link)
* Calculated from the data itself

## Conversion

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

## Outputs

In this example the data files were ingested into a single TileDB sparse array within an AWS bucket. The bucket isn't currently publicly available. At the moment this is purely for demonstrative purposes and not reflective of the actual direction in having publicly accessible data.
The output of the metadata generated using the STAC specification can be found [here](https://github.com/ausseabed/special-potato/blob/main/NIDEM-samples/sample-output/NIDEM_25m.stac.json). The format is JSON, specifically GeoJSON, and can be downloaded and viewed like any other geospatial vector file.
