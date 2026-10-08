# bioio-tomocube-tiff

[![Build Status](https://github.com/bioio-devs/bioio-tomocube-tiff/actions/workflows/ci.yml/badge.svg)](https://github.com/bioio-devs/bioio-tomocube-tiff/actions)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)

A BioIO reader plugin for the multi-file TIFF export of the Tomocube HTX
ProcessingServer. The instrument's native `.TCF` files are read by
[bioio-tomocube](https://github.com/bioio-devs/bioio-tomocube).

---

## Documentation

[See the full documentation on our GitHub pages site](https://bioio-devs.github.io/bioio/OVERVIEW.html) - the generic use and installation instructions there will work for this package.

Information about the base reader this package relies on can be found in the `bioio-base` repository [here](https://github.com/bioio-devs/bioio-base)

## Installation

**Stable Release:** `pip install bioio-tomocube-tiff`<br>
**Development Head:** `pip install git+https://github.com/bioio-devs/bioio-tomocube-tiff.git`

## What it reads

The HTX ProcessingServer writes one multi-page TIFF per modality, channel and
timepoint, with the refractive-index volume beside a folder of fluorescence files:

```
data/
    <base>.TP01_HT3D_0.00.TIFF          # refractive index x 10 000, uint16
    <base>.TP01/
        <base>.TP01_FL3D_CH0_0.00.TIFF  # fluorescence, resampled onto the HT grid
        <base>.TP01_FL3D_CH1_0.00.TIFF
        <base>.TP01_FL2D_CH0_0.00.TIFF  # max projection, when exported
    <base>.TP02_HT3D_0.00.TIFF
    <base>.TP02/
        ...
```

The directory is the image. Files that share a `DateTime` tag form one timepoint
and the distinct tags, in order, are `T`. Pass the directory; a single TIFF inside
it is not supported. The modalities become the channels of one scene:

| Scene | Dims | Channels |
|---|---|---|
| `3D` | `TCZYX` | `HT`, `FL_CH0`, `FL_CH1`, … |
| `2DMIP` | `TCZYX`, `Z=1` | same names, only when FL2D/HT2D files exist |

Both scenes span the same `T`. A `(T, C)` plane with no exported file is read as
zeros.

## Issues
[_Click here to view all open issues in bioio-devs organization at once_](https://github.com/search?q=user%3Abioio-devs+is%3Aissue+is%3Aopen&type=issues&ref=advsearch) or check this repository's issue tab.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.
