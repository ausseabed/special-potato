# HMAS Canberra Example

## Code
See [cbr_convert.py](cbr_convert.py)
See [extract_gsf.py](extract_gsf.py)




## Outputs

In this example the data files were ingested into a single TileDB sparse array within an AWS bucket. The TileDB arrays and STAC metadata are publicly available at [TileDB and STAC examples](https://ausseabed-pl019-baseline-data.s3.amazonaws.com/index.html#tiledb-samples/). At the moment this is purely for demonstrative purposes.
The output of the metadata was generated using the STAC specification. The format is JSON, specifically GeoJSON, and can be downloaded and viewed like any other geospatial vector file.

### Metadata
* STAC output [HMAS-Canberra.stac.json](sample-output/HMAS-Canberra.stac.json)
* [Other](sample-output) metadata outputs generated as part of the processing

### Jupyter Notebook
* [Interactive via NBViewer](https://nbviewer.jupyter.org/github/sixy6e/special-potato/blob/main/HMAS-Canberra-sample/hmas-canberra-metadata-display.ipynb)
* [Static on GitHub](hmas-canberra-metadata-display.ipynb)

## Usage (requires the correct Python environment not detailed here)

```bash
python cbr_convert.py --help
Usage: cbr_convert.py [OPTIONS]

  Data conversion process (GSF -> TileDB) for the sample HMAS Canberra data.
  STAC metadata generation.

Options:
  --uri-name TEXT                The URI for the output location of the TileDB
                                 data file.

  --zip-pathname PATH            The pathname to the zip file containing the
                                 sample NIDEM data.

  --tiledb-config-pathname PATH  The pathname to the TileDB config file.
                                 Required for writing to AWS S3.

  --outdir DIRECTORY             The base output directory for storing local
                                 files.

  --help                         Show this message and exit.
```
