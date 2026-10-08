import pathlib
import shutil

import pytest

LOCAL_RESOURCES_DIR = pathlib.Path(__file__).parent / "resources"


@pytest.fixture
def export_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """A writable copy of the example export for tests that damage it."""
    export = tmp_path / "tiff"
    shutil.copytree(LOCAL_RESOURCES_DIR / "tiff", export)
    return export
