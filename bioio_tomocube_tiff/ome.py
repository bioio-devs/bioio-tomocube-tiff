import numpy as np
from ome_types.model import (
    OME,
    Channel,
    Image,
    Instrument,
    InstrumentRef,
    Microscope,
    Pixels,
    Pixels_DimensionOrder,
    PixelType,
    Plane,
    UnitsTime,
)

from .tomocube_file import TomocubeFile, TomocubeScene

###############################################################################

_PIXEL_TYPES = {"float32": PixelType.FLOAT, "float64": PixelType.DOUBLE}

###############################################################################


def build_ome(file: TomocubeFile) -> OME:
    first = next(iter(file.scenes.values()))
    model = first.channel_files[0].tags.get("Model", "HTX")
    return OME(
        images=[_ome_image(i, s) for i, s in enumerate(file.scenes.values())],
        instruments=[
            Instrument(
                id="Instrument:0",
                microscope=Microscope(manufacturer="Tomocube", model=str(model)),
            )
        ],
    )


def _ome_image(index: int, scene: TomocubeScene) -> Image:
    size_z, size_y, size_x = scene.shape
    missing = set(scene.missing)
    t0 = next((dt for dt in scene.datetimes if dt is not None), None)
    planes = []
    for t, dt in enumerate(scene.datetimes):
        delta = (dt - t0).total_seconds() if dt and t0 else None
        planes += [
            Plane(
                the_t=t, the_c=c, the_z=0, delta_t=delta, delta_t_unit=UnitsTime.SECOND
            )
            for c in range(len(scene.channels))
            if (t, c) not in missing
        ]
    interval = scene.time_interval
    pixel_sizes = scene.pixel_sizes
    dtype_name = np.dtype(scene.dtype).name
    return Image(
        id=f"Image:{index}",
        name=scene.name,
        acquisition_date=t0,
        instrument_ref=InstrumentRef(id="Instrument:0"),
        pixels=Pixels(
            id=f"Pixels:{index}",
            dimension_order=Pixels_DimensionOrder.XYZCT,
            type=_PIXEL_TYPES.get(dtype_name) or PixelType(dtype_name),
            size_t=len(scene.datetimes),
            size_c=len(scene.channels),
            size_z=size_z,
            size_y=size_y,
            size_x=size_x,
            physical_size_x=pixel_sizes.X,
            physical_size_y=pixel_sizes.Y,
            physical_size_z=pixel_sizes.Z,
            time_increment=interval.total_seconds() if interval else None,
            channels=[
                Channel(id=f"Channel:{index}:{c}", name=name, samples_per_pixel=1)
                for c, name in enumerate(scene.channels)
            ],
            planes=planes,
        ),
    )
