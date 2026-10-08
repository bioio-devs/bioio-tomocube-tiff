from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import dask
import dask.array as da
import numpy as np
import tifffile
import xarray as xr
from bioio_base import constants, exceptions, io, reader, types
from bioio_base.dimensions import DimensionNames
from fsspec.spec import AbstractFileSystem
from ome_types.model import OME

from .ome import build_ome
from .tomocube_file import TomocubeFile, TomocubeScene

###############################################################################


def _read_pages(
    fs: AbstractFileSystem, path: str, key: Any, scale: Optional[float], dtype: Any
) -> np.ndarray:
    """Read page(s) *key* of one file, ignoring its ImageJ series description."""
    with fs.open(path, "rb") as f, tifffile.TiffFile(f) as tif:
        data = tif.asarray(key=key)
    if scale is not None:
        data = data * np.float32(scale)
    return data.astype(dtype, copy=False)


###############################################################################


class Reader(reader.Reader):
    """Read TIFF exports written by the Tomocube HTX ProcessingServer.

    The server writes one multi-page TIFF per modality, channel and timepoint::

        <name>.TP01_HT3D_0.00.TIFF                  refractive index x 10 000
        <name>.TP01/<name>.TP01_FL3D_CH0_0.00.TIFF  fluorescence, on the HT grid
        <name>.TP01/<name>.TP01_FL2D_CH0_0.00.TIFF  max projection
        <name>.TP02_HT3D_0.00.TIFF ...

    The directory is the image. Files sharing a ``DateTime`` tag form one
    timepoint, and the distinct tags, in order, are ``T``. The modalities share
    a pixel grid and become channels: scene ``3D`` holds ``HT``, ``FL_CH0``, ...
    as ``TCZYX``; scene ``2DMIP`` holds the projections with ``Z=1`` when they
    were exported. Both scenes span the same ``T``. A ``(T, C)`` plane with no
    file is read as zeros.

    Parameters
    ----------
    image : types.PathLike
        The export directory.
    fs_kwargs : Dict[str, Any]
        Passed to fsspec when constructing the filesystem.
    """

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        paths = TomocubeFile.discover(fs, path)
        with fs.open(paths[0], "rb") as f, tifffile.TiffFile(f) as tif:
            tags = tif.pages[0].tags
            ident = f"{tags.valueof('Model', '')} {tags.valueof('Software', '')}"
        if "HTX" not in ident.upper() and "TOMO" not in ident.upper():
            raise exceptions.UnsupportedFileFormatError(
                "bioio-tomocube-tiff",
                path,
                f"Model/Software tags {ident!r} do not identify a Tomocube export.",
            )
        return True

    def __init__(
        self,
        image: types.PathLike,
        fs_kwargs: Dict[str, Any] = {},
    ):
        self._fs, self._path = io.pathlike_to_fs(
            image, enforce_exists=True, fs_kwargs=fs_kwargs
        )
        self._is_supported_image(self._fs, self._path)
        self._file = TomocubeFile(self._fs, self._path)
        self._scenes = tuple(self._file.scenes)
        self._current_scene_index = 0

    @property
    def scenes(self) -> Tuple[str, ...]:
        return self._scenes

    @property
    def _scene(self) -> TomocubeScene:
        return self._file.scenes[self.current_scene]

    def _read_delayed(self) -> xr.DataArray:
        scene = self._scene
        size_z, size_y, size_x = scene.shape
        timepoints = []
        for row in scene.files:
            channels = []
            for file, channel in zip(row, scene.channels):
                if file is None:
                    channels.append(
                        da.zeros(
                            scene.shape, chunks=(1, size_y, size_x), dtype=scene.dtype
                        )
                    )
                    continue
                read = dask.delayed(_read_pages)
                channels.append(
                    da.stack(
                        [
                            da.from_delayed(
                                read(
                                    self._fs,
                                    file.path,
                                    z,
                                    scene.scale(channel),
                                    scene.dtype,
                                ),
                                shape=(size_y, size_x),
                                dtype=scene.dtype,
                            )
                            for z in range(size_z)
                        ]
                    )
                )
            timepoints.append(da.stack(channels))
        return self._to_xarray(da.stack(timepoints))

    def _read_immediate(self) -> xr.DataArray:
        scene = self._scene
        timepoints = []
        for row in scene.files:
            channels = []
            for file, channel in zip(row, scene.channels):
                if file is None:
                    channels.append(np.zeros(scene.shape, dtype=scene.dtype))
                    continue
                pages = _read_pages(
                    self._fs,
                    file.path,
                    slice(None),
                    scene.scale(channel),
                    scene.dtype,
                )
                channels.append(pages.reshape(scene.shape))
            timepoints.append(np.stack(channels))
        return self._to_xarray(np.stack(timepoints))

    def _to_xarray(self, data: Any) -> xr.DataArray:
        return xr.DataArray(
            data,
            dims=[
                DimensionNames.Time,
                DimensionNames.Channel,
                DimensionNames.SpatialZ,
                DimensionNames.SpatialY,
                DimensionNames.SpatialX,
            ],
            coords={DimensionNames.Channel: list(self._scene.channels)},
            attrs={constants.METADATA_PROCESSED: self.ome_metadata},
        )

    @property
    def dtype(self) -> np.dtype:
        return self._scene.dtype

    @property
    def channel_names(self) -> Optional[List[str]]:
        return list(self._scene.channels)

    @property
    def physical_pixel_sizes(self) -> types.PhysicalPixelSizes:
        return self._scene.pixel_sizes

    @property
    def time_interval(self) -> Optional[timedelta]:
        return self._scene.time_interval

    @property
    def ome_metadata(self) -> OME:
        return build_ome(self._file)
