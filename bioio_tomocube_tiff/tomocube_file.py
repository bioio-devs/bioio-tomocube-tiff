import logging
import posixpath
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import tifffile
from bioio_base import exceptions, types
from fsspec.spec import AbstractFileSystem

###############################################################################

log = logging.getLogger(__name__)

#: ``HT3D``/``HT2D`` files store refractive index x 10 000 as uint16.
RI_SCALE = 1.0e-4

# The modality token of an export file name, e.g. "_HT3D_" or "_FL3D_CH0_".
MODALITY_RE = re.compile(
    r"_(?P<modality>HT|FL)(?P<dims>3D|2D)(?:_CH(?P<channel>\d+))?_(?!CH\d+[._])"
    r".*\.tiff?$",
    re.IGNORECASE,
)
_RESUNIT_TO_UM = {2: 25_400.0, 3: 10_000.0}
_IMAGEJ_UNIT_TO_UM = {"cm": 1e4, "mm": 1e3, "um": 1.0, "µm": 1.0, "micron": 1.0}
_TAGS_TO_SKIP = {"StripOffsets", "StripByteCounts"}

###############################################################################


class _Tiff(NamedTuple):
    """One exported TIFF: the modality from its name and what its header says."""

    path: str
    modality: str  # HT or FL
    scene: str  # 3D or 2DMIP
    channel: Optional[int]
    shape: Tuple[int, int, int]  # pages, Y, X
    dtype: np.dtype
    datetime: Optional[datetime]
    pixel_size_yx: Tuple[Optional[float], Optional[float]]
    z_spacing: Optional[float]
    tags: Dict[str, Any]
    imagej: Dict[str, Any]

    @classmethod
    def open(cls, fs: AbstractFileSystem, path: str) -> "_Tiff":
        """Parse the file name and read the first page's tags."""
        match = MODALITY_RE.search(posixpath.basename(path))
        if match is None:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-tomocube-tiff",
                path,
                "File name has no Tomocube modality token (_HT3D_, _FL3D_CHn_, ...).",
            )
        with fs.open(path, "rb") as f, tifffile.TiffFile(f) as tif:
            page = tif.pages[0]
            n_pages = len(tif.pages)
            shape, dtype = page.shape, page.dtype
            res_x, res_y = page.resolution
            unit = _RESUNIT_TO_UM.get(int(page.resolutionunit))
            tags = {t.name: t.value for t in page.tags if t.name not in _TAGS_TO_SKIP}
            imagej = dict(tif.imagej_metadata or {})
        z_unit = _IMAGEJ_UNIT_TO_UM.get(str(imagej.get("unit", "um")).lower())
        channel = match.group("channel")
        return cls(
            path=path,
            modality=match.group("modality").upper(),
            scene="3D" if match.group("dims").upper() == "3D" else "2DMIP",
            channel=int(channel) if channel is not None else None,
            shape=(n_pages, *shape),
            dtype=np.dtype(dtype),
            datetime=(
                datetime.strptime(tags["DateTime"], "%Y:%m:%d %H:%M:%S")
                if "DateTime" in tags
                else None
            ),
            pixel_size_yx=(
                unit / res_y if unit and res_y else None,
                unit / res_x if unit and res_x else None,
            ),
            z_spacing=(
                imagej["spacing"] * z_unit if "spacing" in imagej and z_unit else None
            ),
            tags=tags,
            imagej=imagej,
        )

    @property
    def channel_name(self) -> str:
        return "HT" if self.modality == "HT" else f"FL_CH{self.channel or 0}"

    @property
    def sort_key(self) -> Tuple[bool, int]:
        """HT before the FL channels by index."""
        return (self.modality == "FL", self.channel or 0)


class TomocubeScene(NamedTuple):
    """One scene of an export: a ``[T][C]`` grid of TIFFs on one pixel grid."""

    name: str
    channels: Tuple[str, ...]
    datetimes: Tuple[Optional[datetime], ...]
    files: Tuple[Tuple[Optional[_Tiff], ...], ...]

    @property
    def channel_files(self) -> Tuple[_Tiff, ...]:
        """First exported file of each channel."""
        return tuple(
            [f for f in column if f is not None][0] for column in zip(*self.files)
        )

    @property
    def missing(self) -> List[Tuple[int, int]]:
        """``(T, C)`` planes with no exported file; read as zeros."""
        return [
            (t, c)
            for t, row in enumerate(self.files)
            for c, f in enumerate(row)
            if f is None
        ]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.channel_files[0].shape

    @property
    def dtype(self) -> np.dtype:
        if "HT" in self.channels:
            return np.dtype(np.float32)
        return self.channel_files[0].dtype

    def scale(self, channel: str) -> Optional[float]:
        return RI_SCALE if channel == "HT" else None

    @property
    def pixel_sizes(self) -> types.PhysicalPixelSizes:
        files = self.channel_files
        # FL3D headers keep the pre-resampling Z step; the pages are on the HT grid.
        ht_first = sorted(files, key=lambda f: f.modality != "HT")
        z = next((f.z_spacing for f in ht_first if f.z_spacing), None)
        return types.PhysicalPixelSizes(
            z if self.shape[0] > 1 else None, *files[0].pixel_size_yx
        )

    @property
    def time_interval(self) -> Optional[timedelta]:
        first, last = self.datetimes[0], self.datetimes[-1]
        if len(self.datetimes) < 2 or first is None or last is None:
            return None
        return (last - first) / (len(self.datetimes) - 1)


def _build_scenes(files: Sequence[_Tiff]) -> Dict[str, TomocubeScene]:
    """Group TIFFs into scenes sharing one ``T`` axis and one pixel grid."""
    datetimes = tuple(
        sorted(
            {f.datetime for f in files}, key=lambda d: (d is None, d or datetime.min)
        )
    )
    scenes: Dict[str, TomocubeScene] = {}
    for name in ("3D", "2DMIP"):
        members = sorted(
            (f for f in files if f.scene == name), key=lambda f: f.sort_key
        )
        if not members:
            continue
        grids = {(f.shape, f.dtype.str) for f in members}
        if len(grids) > 1:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-tomocube-tiff",
                members[0].path,
                f"Scene {name}: files are not on one pixel grid: {sorted(grids)}.",
            )
        channels = tuple(dict.fromkeys(f.channel_name for f in members))
        lookup = {(f.datetime, f.channel_name): f for f in members}
        scene = TomocubeScene(
            name=name,
            channels=channels,
            datetimes=datetimes,
            files=tuple(
                tuple(lookup.get((dt, c)) for c in channels) for dt in datetimes
            ),
        )
        if scene.missing:
            log.warning(
                "Scene %s: no file exported for (T, C) planes %s; reading them as "
                "zeros.",
                name,
                scene.missing,
            )
        scenes[name] = scene
    return scenes


###############################################################################


class TomocubeFile:
    """An HTX export, opened from its directory or from any TIFF inside it.

    Parameters
    ----------
    fs, path:
        The export directory.

    Attributes
    ----------
    scenes:
        ``"3D"`` and, when projections were exported, ``"2DMIP"``.
    """

    def __init__(
        self,
        fs: AbstractFileSystem,
        path: str,
    ):
        files = [_Tiff.open(fs, p) for p in self.discover(fs, path)]
        self.scenes: Dict[str, TomocubeScene] = _build_scenes(files)

    @staticmethod
    def discover(fs: AbstractFileSystem, path: str) -> List[str]:
        """Paths of every export TIFF in the export directory *path*."""
        if not fs.isdir(path):
            raise exceptions.UnsupportedFileFormatError(
                "bioio-tomocube-tiff", path, "Not a directory."
            )

        entries = fs.ls(path, detail=True)
        names = [e["name"] for e in entries if e["type"] != "directory"]
        for e in entries:
            if e["type"] == "directory":
                names += fs.ls(e["name"], detail=False)
        paths = [n for n in names if MODALITY_RE.search(posixpath.basename(n))]
        if not paths:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-tomocube-tiff",
                path,
                f"No Tomocube export TIFFs found in {path!r}.",
            )
        return sorted(paths)
