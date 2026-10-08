import shutil
from datetime import datetime, timedelta

import dask.array as da
import fsspec
import numpy as np
import pytest
import tifffile
from bioio import BioImage
from bioio_base import exceptions, test_utilities
from ome_types.model import OME

from bioio_tomocube_tiff import Reader
from bioio_tomocube_tiff.tomocube_file import RI_SCALE, TomocubeFile

from .conftest import LOCAL_RESOURCES_DIR

TIFF_DIR = LOCAL_RESOURCES_DIR / "tiff"
BASE = "251003.112316.6 Well Mito.018.Group1.A1"
HT_TP01 = TIFF_DIR / f"{BASE}.TP01_HT3D_0.00.TIFF"
HT_TP05 = TIFF_DIR / f"{BASE}.TP05_HT3D_0.00.TIFF"
FL3D_CH0_TP01 = TIFF_DIR / f"{BASE}.TP01" / f"{BASE}.TP01_FL3D_CH0_0.00.TIFF"
FL3D_CH1_TP01 = TIFF_DIR / f"{BASE}.TP01" / f"{BASE}.TP01_FL3D_CH1_0.00.TIFF"
FL3D_CH0_TP05 = TIFF_DIR / f"{BASE}.TP05" / f"{BASE}.TP05_FL3D_CH0_0.00.TIFF"
FL3D_CH1_TP05 = TIFF_DIR / f"{BASE}.TP05" / f"{BASE}.TP05_FL3D_CH1_0.00.TIFF"
FL2D_CH0_TP05 = TIFF_DIR / f"{BASE}.TP05" / f"{BASE}.TP05_FL2D_CH0_0.00.TIFF"
FL2D_CH1_TP05 = TIFF_DIR / f"{BASE}.TP05" / f"{BASE}.TP05_FL2D_CH1_0.00.TIFF"

PX_XY = 0.1625607173804458  # 128 / 7873981 cm
PX_Z = 0.87449127  # HT ImageJ spacing, 0.000087449127 cm
CHANNELS_3D = ["HT", "FL_CH0", "FL_CH1"]
CHANNELS_2D = ["FL_CH0", "FL_CH1"]
SCENES = ("3D", "2DMIP")
SHAPE_3D = (2, 3, 70, 32, 32)
SHAPE_2D = (2, 2, 1, 32, 32)
INTERVAL = timedelta(seconds=354)  # TP01 12:36:59 -> TP05 12:42:53
T0, T1 = datetime(2025, 10, 3, 12, 36, 59), datetime(2025, 10, 3, 12, 42, 53)
SERIES = {"3D": CHANNELS_3D, "2DMIP": CHANNELS_2D}


def _copy_fixtures(root, skip=()):
    """Mirror the fixture folder into *root*."""
    for src in TIFF_DIR.rglob("*.TIFF"):
        if src.name not in skip:
            dst = root / src.relative_to(TIFF_DIR)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)


def _pages(path):
    with tifffile.TiffFile(path) as tif:
        return np.stack([p.asarray() for p in tif.pages])


def _summary(path):
    """scene -> channels, and the T axis, of the export *path* belongs to."""
    scenes = TomocubeFile(fsspec.filesystem("file"), str(path)).scenes
    return {name: list(s.channels) for name, s in scenes.items()}, {
        s.datetimes for s in scenes.values()
    }


# ---------------------------------------------------------------------------
# File names and discovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        (f"{BASE}.TP01_HT3D_0.00.TIFF", {"3D": ["HT"]}),
        (f"{BASE}.TP05_FL3D_CH1_0.00.TIFF", {"3D": ["FL_CH1"]}),
        (f"{BASE}.TP05_FL2D_CH0_0.00.TIFF", {"2DMIP": ["FL_CH0"]}),
        ("exp_HT3D_0.00.tif", {"3D": ["HT"]}),
        ("exp.A1.TP12_HT2D_1.50.TIFF", {"2DMIP": ["HT"]}),
    ],
)
def test_parse_filename(tmp_path, name, expected):
    """Only the modality token of the name is read; the rest comes from tags."""
    shutil.copyfile(HT_TP01, tmp_path / name)
    assert _summary(tmp_path / name) == (expected, {(T0,)})


@pytest.mark.parametrize(
    "name", ["image.tiff", "HT3D.TIFF", f"{BASE}.TP01_BF_0.00.TIFF", "x_FL3D_CH0.TIFF"]
)
def test_parse_filename_rejects(tmp_path, name):
    shutil.copyfile(HT_TP01, tmp_path / name)
    with pytest.raises(exceptions.UnsupportedFileFormatError, match="modality token"):
        Reader(tmp_path / name)


@pytest.mark.parametrize("anchor", [HT_TP01, FL3D_CH1_TP05, FL2D_CH0_TP05, TIFF_DIR])
def test_discover_full_series(anchor):
    """Any file, even one inside a TP folder, or the directory finds every file."""
    assert _summary(anchor) == (SERIES, {(T0, T1)})


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, set_scene, expected_shape, expected_channel_names, "
    "expected_physical_pixel_sizes",
    [
        (HT_TP01, "3D", SHAPE_3D, CHANNELS_3D, (PX_Z, PX_XY, PX_XY)),
        (FL3D_CH1_TP05, "3D", SHAPE_3D, CHANNELS_3D, (PX_Z, PX_XY, PX_XY)),
        (FL2D_CH0_TP05, "2DMIP", SHAPE_2D, CHANNELS_2D, (None, PX_XY, PX_XY)),
        (HT_TP05, "2DMIP", SHAPE_2D, CHANNELS_2D, (None, PX_XY, PX_XY)),
        (TIFF_DIR, "3D", SHAPE_3D, CHANNELS_3D, (PX_Z, PX_XY, PX_XY)),
        (TIFF_DIR, "2DMIP", SHAPE_2D, CHANNELS_2D, (None, PX_XY, PX_XY)),
    ],
)
def test_reader(
    filename,
    set_scene,
    expected_shape,
    expected_channel_names,
    expected_physical_pixel_sizes,
):
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=filename,
        set_scene=set_scene,
        expected_scenes=SCENES,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=np.dtype(np.float32 if set_scene == "3D" else np.uint16),
        expected_dims_order="TCZYX",
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=OME,
        reader_kwargs={},
    )


def test_multi_scene():
    test_utilities.run_multi_scene_image_read_checks(
        ImageContainer=Reader,
        image=HT_TP05,
        first_scene_id="3D",
        first_scene_shape=SHAPE_3D,
        first_scene_dtype=np.dtype(np.float32),
        second_scene_id="2DMIP",
        second_scene_shape=SHAPE_2D,
        second_scene_dtype=np.dtype(np.uint16),
        allow_same_scene_data=False,
        reader_kwargs={},
    )


@pytest.mark.parametrize(
    "filename, expected_scene",
    [
        (HT_TP01, "3D"),
        (FL3D_CH1_TP05, "3D"),
        (FL2D_CH0_TP05, "2DMIP"),
        (TIFF_DIR, "3D"),
    ],
)
def test_default_scene_matches_anchor_file(filename, expected_scene):
    assert Reader(filename).current_scene == expected_scene


def test_unsupported(tmp_path):
    with pytest.raises(exceptions.UnsupportedFileFormatError):
        Reader(LOCAL_RESOURCES_DIR / "unsupported.tif")
    # a matching name without HTX tags
    tifffile.imwrite(tmp_path / "x.TP01_HT3D_0.00.TIFF", np.zeros((4, 4), np.uint16))
    with pytest.raises(exceptions.UnsupportedFileFormatError, match="Model/Software"):
        Reader(tmp_path / "x.TP01_HT3D_0.00.TIFF")
    # a directory without export files
    (tmp_path / "empty").mkdir()
    with pytest.raises(exceptions.UnsupportedFileFormatError, match="No Tomocube"):
        Reader(tmp_path / "empty")


def test_mismatched_grids_rejected(tmp_path):
    _copy_fixtures(tmp_path, skip={FL3D_CH0_TP01.name})
    with tifffile.TiffFile(FL3D_CH0_TP01) as tif:
        pages = np.stack([p.asarray() for p in tif.pages[:10]])
    tifffile.imwrite(
        tmp_path / f"{BASE}.TP01" / FL3D_CH0_TP01.name,
        pages,
        extratags=[(272, "s", 0, "HTX", True)],
    )
    with pytest.raises(exceptions.UnsupportedFileFormatError, match="pixel grid"):
        Reader(tmp_path / HT_TP01.name)


# ---------------------------------------------------------------------------
# Pixel data
# ---------------------------------------------------------------------------


def test_channels_and_timepoints_come_from_the_right_files():
    rdr = Reader(TIFF_DIR)
    data = rdr.data
    assert data.shape == SHAPE_3D and rdr.channel_names == CHANNELS_3D
    for t, c, path in [
        (0, 0, HT_TP01),
        (0, 1, FL3D_CH0_TP01),
        (0, 2, FL3D_CH1_TP01),
        (1, 0, HT_TP05),
        (1, 1, FL3D_CH0_TP05),
        (1, 2, FL3D_CH1_TP05),
    ]:
        expected = _pages(path)
        if c == 0:
            expected = expected.astype(np.float32) * RI_SCALE
        np.testing.assert_array_equal(data[t, c], expected)
    np.testing.assert_array_equal(rdr.dask_data.compute(), data)


def test_projection_scene_is_padded_to_full_t_axis():
    rdr = Reader(TIFF_DIR)
    rdr.set_scene("2DMIP")
    assert rdr.shape == SHAPE_2D
    for data in (rdr.data, rdr.dask_data.compute()):
        assert not data[0].any()
        np.testing.assert_array_equal(data[1, 0], _pages(FL2D_CH0_TP05))
        np.testing.assert_array_equal(data[1, 1], _pages(FL2D_CH1_TP05))


def test_fl3d_reads_all_pages_not_imagej_slices():
    with tifffile.TiffFile(FL3D_CH0_TP01) as tif:
        assert tif.imagej_metadata["slices"] == 12 and len(tif.pages) == 70
    fl0 = Reader(FL3D_CH0_TP01).data[0, 1]
    assert fl0.shape == (70, 32, 32)
    nonzero = [z for z in range(70) if fl0[z].any()]
    assert nonzero[0] == 30 and nonzero[-1] == 43 and len(nonzero) == 14


def test_z_spacing_comes_from_ht_header():
    rdr = Reader(FL3D_CH0_TP01)
    assert rdr.physical_pixel_sizes.Z == pytest.approx(PX_Z)


def test_ht_is_refractive_index():
    raw = _pages(HT_TP01)
    assert raw.dtype == np.uint16
    rdr = Reader(HT_TP01)
    data = rdr.data
    assert rdr.dtype == data.dtype == np.float32
    np.testing.assert_allclose(data[0, 0], raw.astype(np.float32) * RI_SCALE)
    assert 1.30 < data[:, 0].min() < data[:, 0].max() < 1.45
    assert rdr.ome_metadata.images[0].pixels.type.value == "float"


def test_dask_chunks_one_plane_each():
    lazy = Reader(HT_TP01).xarray_dask_data
    assert isinstance(lazy.data, da.Array) and lazy.dims == ("T", "C", "Z", "Y", "X")
    assert lazy.data.chunks[:3] == ((1, 1), (1, 1, 1), (1,) * 70)
    assert lazy[1, 2, 35].compute().shape == (32, 32)


def test_incomplete_timepoint_is_padded(tmp_path, caplog):
    _copy_fixtures(tmp_path, skip={FL3D_CH1_TP05.name})
    with caplog.at_level("WARNING", logger="bioio_tomocube_tiff.reader"):
        rdr = Reader(tmp_path / HT_TP01.name)
    assert rdr.shape == SHAPE_3D
    data = rdr.data
    assert not data[1, 2].any()
    np.testing.assert_array_equal(data[0, 2], _pages(FL3D_CH1_TP01))
    planes = rdr.ome_metadata.images[0].pixels.planes
    assert len(planes) == 5 and (1, 2) not in {(p.the_t, p.the_c) for p in planes}
    assert any("(T, C) planes [(1, 2)]" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_time_interval_and_ome_planes():
    rdr = Reader(HT_TP01)
    assert rdr.time_interval == INTERVAL
    img3d, img2d = rdr.ome_metadata.images
    assert [(p.the_t, p.the_c, p.delta_t) for p in img3d.pixels.planes] == [
        (0, 0, 0.0),
        (0, 1, 0.0),
        (0, 2, 0.0),
        (1, 0, 354.0),
        (1, 1, 354.0),
        (1, 2, 354.0),
    ]
    # the padded scene shares the T axis and timing; missing cells have no plane
    assert [(p.the_t, p.the_c) for p in img2d.pixels.planes] == [(1, 0), (1, 1)]
    assert img2d.pixels.size_t == 2 and img2d.pixels.time_increment == 354.0
    rdr.set_scene("2DMIP")
    assert rdr.time_interval == INTERVAL


def test_ome_metadata():
    rdr = Reader(HT_TP05)
    ome = rdr.ome_metadata
    assert isinstance(ome, OME) and len(ome.images) == 2
    img3d, img2d = ome.images
    assert (img3d.name, img2d.name) == SCENES
    assert [c.name for c in img3d.pixels.channels] == CHANNELS_3D
    assert [c.name for c in img2d.pixels.channels] == CHANNELS_2D
    assert img2d.pixels.size_z == 1 and img2d.pixels.physical_size_z is None
    assert img3d.pixels.physical_size_z == pytest.approx(PX_Z)
    assert img3d.pixels.physical_size_x == pytest.approx(PX_XY)
    assert img3d.acquisition_date == T0
    assert ome.instruments[0].microscope.model == "HTX"


def test_standard_metadata():
    sm = Reader(FL3D_CH0_TP01).standard_metadata.to_dict()
    expected = {
        "Dimensions Present": "TCZYX",
        "Image Size T": 2,
        "Image Size C": 3,
        "Image Size Z": 70,
        "Image Size Y": 32,
        "Image Size X": 32,
        "Imaging Datetime": T0,
        "Pixel Size Z": pytest.approx(PX_Z),
        "Pixel Size Y": pytest.approx(PX_XY),
        "Pixel Size X": pytest.approx(PX_XY),
        "Timelapse": True,
        "Timelapse Interval": INTERVAL,
    }
    assert {k: sm[k] for k in expected} == expected


# ---------------------------------------------------------------------------
# BioImage and remote filesystems
# ---------------------------------------------------------------------------


def test_bioimage():
    img = BioImage(HT_TP01)
    assert isinstance(img.reader, Reader)
    assert img.shape == SHAPE_3D and img.channel_names == CHANNELS_3D
    np.testing.assert_array_equal(
        img.get_image_data("ZYX", T=1, C=1), _pages(FL3D_CH0_TP05)
    )
    img = BioImage(HT_TP01)
    assert img.dtype == np.float32


def test_bioimage_directory_needs_explicit_reader():
    img = BioImage(TIFF_DIR, reader=Reader)
    assert img.shape == SHAPE_3D
    img.set_scene("2DMIP")
    assert img.shape == SHAPE_2D
    with pytest.raises(exceptions.UnsupportedFileFormatError):
        BioImage(TIFF_DIR)


@pytest.fixture(scope="module")
def mem_fs():
    """The fixture folder mirrored into memory://."""
    mem = fsspec.filesystem("memory")
    for local in TIFF_DIR.rglob("*.TIFF"):
        mem.pipe(
            f"/tomocube/{local.relative_to(TIFF_DIR).as_posix()}", local.read_bytes()
        )
    yield "memory:///tomocube"
    mem.rm("/tomocube", recursive=True)


@pytest.mark.parametrize("relative", ["", f"/{HT_TP05.name}"])
def test_remote_read(mem_fs, relative):
    rdr = Reader(mem_fs + relative)
    assert rdr.scenes == SCENES and rdr.shape == SHAPE_3D
    assert rdr.physical_pixel_sizes.Z == pytest.approx(PX_Z)
    frame = rdr.xarray_dask_data[1, 2, 40].compute()
    np.testing.assert_array_equal(frame, _pages(FL3D_CH1_TP05)[40])
    rdr.set_scene("2DMIP")
    assert rdr.shape == SHAPE_2D
