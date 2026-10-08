from typing import List

import bioio_base.reader_metadata

###############################################################################


class ReaderMetadata(bioio_base.reader_metadata.ReaderMetadata):
    """Plugin-level metadata: the export directory is the image."""

    @staticmethod
    def get_supported_extensions() -> List[str]:
        return []

    @staticmethod
    def supports_directory_images() -> bool:
        return True

    @staticmethod
    def get_reader() -> bioio_base.reader.Reader:
        from .reader import Reader

        return Reader
