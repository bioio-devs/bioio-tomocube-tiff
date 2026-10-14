import pathlib
import typing
from datetime import datetime, timedelta

import dask.array as da
import fsspec
import numpy as np
import pytest
import tifffile
from bioio import BioImage
from bioio_base import dimensions, exceptions, test_utilities
from ome_types.model import OME

from bioio_tomocube_tiff import Reader

from .conftest import LOCAL_RESOURCES_DIR

TIFF_DIR = LOCAL_RESOURCES_DIR / "tiff"


@pytest.mark.parametrize(
    "filename, expected_message",
    [
        ("unsupported.tif", "Not a directory"),
        ("untagged", "Model/Software"),
    ],
)
def test_reader_with_unsupported_inputs(filename: str, expected_message: str) -> None:
    with pytest.raises(exceptions.UnsupportedFileFormatError, match=expected_message):
        Reader(LOCAL_RESOURCES_DIR / filename)


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names, "
    "expected_physical_pixel_sizes",
    [
        (
            "tiff",
            "3D",
            ("3D", "2DMIP"),
            (2, 3, 70, 32, 32),
            np.float32,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["HT", "FL_CH0", "FL_CH1"],
            (0.87449127, 0.1625607173804458, 0.1625607173804458),
        ),
        (
            "tiff",
            "2DMIP",
            ("3D", "2DMIP"),
            (2, 2, 1, 32, 32),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["FL_CH0", "FL_CH1"],
            (None, 0.1625607173804458, 0.1625607173804458),
        ),
    ],
)
def test_reader(
    filename: str,
    set_scene: str,
    expected_scenes: typing.Tuple[str, ...],
    expected_shape: typing.Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: typing.List[str],
    expected_physical_pixel_sizes: typing.Tuple[float, float, float],
) -> None:
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Run checks
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=np.dtype(expected_dtype),
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=OME,
    )


def test_reader_multi_scene() -> None:
    test_utilities.run_multi_scene_image_read_checks(
        ImageContainer=Reader,
        image=TIFF_DIR,
        first_scene_id="3D",
        first_scene_shape=(2, 3, 70, 32, 32),
        first_scene_dtype=np.dtype(np.float32),
        second_scene_id="2DMIP",
        second_scene_shape=(2, 2, 1, 32, 32),
        second_scene_dtype=np.dtype(np.uint16),
        allow_same_scene_data=False,
    )


def test_dask_chunks_one_plane_each() -> None:
    lazy = Reader(TIFF_DIR).xarray_dask_data
    assert isinstance(lazy.data, da.Array)
    assert lazy.data.chunks[:3] == ((1, 1), (1, 1, 1), (1,) * 70)


def test_metadata() -> None:
    rdr = Reader(TIFF_DIR)
    interval = timedelta(seconds=354)
    assert rdr.time_interval == interval

    ome = rdr.ome_metadata
    img3d, img2d = ome.images
    assert [i.name for i in ome.images] == ["3D", "2DMIP"]
    assert img3d.acquisition_date == datetime(2025, 10, 3, 12, 36, 59)
    assert ome.instruments[0].microscope.model == "HTX"
    assert img3d.pixels.type.value == "float"
    assert [(p.the_t, p.the_c, p.delta_t) for p in img3d.pixels.planes] == [
        (0, 0, 0.0),
        (0, 1, 0.0),
        (0, 2, 0.0),
        (1, 0, 354.0),
        (1, 1, 354.0),
        (1, 2, 354.0),
    ]
    # the projections share the T axis; the timepoint without them has no planes
    assert [(p.the_t, p.the_c) for p in img2d.pixels.planes] == [(1, 0), (1, 1)]
    assert img2d.pixels.size_t == 2 and img2d.pixels.time_increment == 354.0

    sm = rdr.standard_metadata.to_dict()
    assert sm["Timelapse"] and sm["Timelapse Interval"] == interval
    assert sm["Imaging Datetime"] == datetime(2025, 10, 3, 12, 36, 59)


def test_incomplete_timepoint_is_padded(
    export_dir: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    next(export_dir.rglob("*TP05_FL3D_CH1*")).unlink()
    with caplog.at_level("WARNING", logger="bioio_tomocube_tiff.tomocube_file"):
        rdr = Reader(export_dir)
    data = rdr.data
    assert data.shape == (2, 3, 70, 32, 32)
    assert not data[1, 2].any() and data[0, 2].any()
    planes = rdr.ome_metadata.images[0].pixels.planes
    assert len(planes) == 5 and (1, 2) not in {(p.the_t, p.the_c) for p in planes}
    assert any("(T, C) planes [(1, 2)]" in r.message for r in caplog.records)


def test_mismatched_grids_rejected(export_dir: pathlib.Path) -> None:
    fl = next(export_dir.rglob("*TP01_FL3D_CH0*"))
    with tifffile.TiffFile(fl) as tif:
        pages = np.stack([p.asarray() for p in tif.pages[:10]])
    tifffile.imwrite(fl, pages, extratags=[(272, "s", 0, "HTX", True)])
    with pytest.raises(exceptions.UnsupportedFileFormatError, match="pixel grid"):
        Reader(export_dir)


def test_files_without_modality_token_are_ignored(export_dir: pathlib.Path) -> None:
    for i, tiff in enumerate(list(export_dir.rglob("*.TIFF"))):
        tiff.rename(tiff.with_name(f"image{i}.tiff"))
    with pytest.raises(exceptions.UnsupportedFileFormatError, match="No Tomocube"):
        Reader(export_dir)


def test_bioimage() -> None:
    img = BioImage(TIFF_DIR, reader=Reader)
    assert isinstance(img.reader, Reader)
    assert img.shape == (2, 3, 70, 32, 32)
    assert img.channel_names == ["HT", "FL_CH0", "FL_CH1"]
    assert img.dtype == np.float32
    img.set_scene("2DMIP")
    assert img.shape == (2, 2, 1, 32, 32)


@pytest.fixture(scope="module")
def mem_fs() -> typing.Iterator[str]:
    """The example export mirrored into memory://."""
    mem = fsspec.filesystem("memory")
    for local in TIFF_DIR.rglob("*.TIFF"):
        mem.pipe(
            f"/tomocube/{local.relative_to(TIFF_DIR).as_posix()}", local.read_bytes()
        )
    yield "memory:///tomocube"
    mem.rm("/tomocube", recursive=True)


def test_remote_read(mem_fs: str) -> None:
    rdr = Reader(mem_fs)
    assert rdr.scenes == ("3D", "2DMIP")
    assert rdr.shape == (2, 3, 70, 32, 32)
    assert rdr.xarray_dask_data[1, 2, 40].compute().shape == (32, 32)
