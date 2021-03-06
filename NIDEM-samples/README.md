# NIDEM

See https://cmi.ga.gov.au/data-products/dea/325/dea-intertidal-elevation-landsat for specifics on NIDEM.
The metadata was harvested from varied sources:
* Supplied Excel Spreasheet
* CMI (see above link)
* Calculated from the data itself

## Code
See [nidem_convert.py](nidem_convert.py)

## Outputs

In this example the data files were ingested into a single TileDB sparse array within an AWS bucket. The TileDB arrays and STAC metadata are publicly available at [TileDB and STAC examples](https://ausseabed-pl019-baseline-data.s3.amazonaws.com/index.html#tiledb-samples/). At the moment this is purely for demonstrative purposes.
The output of the metadata was generated using the STAC specification. The format is JSON, specifically GeoJSON, and can be downloaded and viewed like any other geospatial vector file.

### Metadata
* STAC output [NIDEM_25m.stac.json](sample-output/NIDEM_25m.stac.json)
* [Other](sample-output) metadata outputs generated as part of the processing

### Jupyter Notebook
* [Interactive via NBViewer](https://nbviewer.jupyter.org/github/ausseabed/special-potato/blob/main/NIDEM-samples/nidem-metadata-display.ipynb)
* [Static on GitHub](nidem-metadata-display.ipynb)

## Usage (requires the correct Python environment not detailed here)

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
