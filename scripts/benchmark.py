"""
    Runs bioio_base's benchmark function against the test resources in this repository
"""

import pathlib

import bioio_base.benchmark

import bioio_tomocube_tiff

# This file is under /scripts while the test resources are under
# /bioio_tomocube_tiff/tests/resources. The export directory is the image.
test_resources_dir = (
    pathlib.Path(__file__).parent.parent / "bioio_tomocube_tiff" / "tests" / "resources"
)
test_files = [test_resources_dir / "tiff"]
print(f"Test files: {[file.name for file in test_files]}")
bioio_base.benchmark.benchmark(bioio_tomocube_tiff.Reader, test_files)
