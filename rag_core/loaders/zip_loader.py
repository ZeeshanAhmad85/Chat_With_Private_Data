"""
Secure ZIP archive loader.
Protects against zip bombs and directory traversal, returning flat lists of Documents.
"""

import io
from pathlib import Path
from typing import BinaryIO, Callable, Dict, List, Union
import zipfile
from rag_core.models import Document


class ZipLoader:
    """Safely extracts and unpacks supported documents from ZIP archives."""

    MAX_UNCOMPRESSED_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB max unpacked

    def __init__(self, loader_factory_fn: Callable[[str], object]):
        self.loader_factory_fn = loader_factory_fn

    def load(self, source: Union[str, Path, BinaryIO], filename: str) -> List[Document]:
        if isinstance(source, (str, Path)):
            zip_stream = open(source, "rb")
            close_stream = True
        elif hasattr(source, "read"):
            content = source.read()
            if hasattr(source, "seek"):
                source.seek(0)
            zip_stream = io.BytesIO(content)
            close_stream = False
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        try:
            with zipfile.ZipFile(zip_stream, "r") as archive:
                total_uncompressed_bytes = 0
                for info in archive.infolist():
                    total_uncompressed_bytes += info.file_size
                    if total_uncompressed_bytes > self.MAX_UNCOMPRESSED_SIZE_BYTES:
                        raise ValueError(
                            f"ZIP archive '{filename}' exceeds maximum allowed uncompressed size "
                            f"({self.MAX_UNCOMPRESSED_SIZE_BYTES / (1024*1024):.0f} MB)."
                        )

                documents: List[Document] = []
                extracted_names = []

                for member in archive.namelist():
                    # Defend against zip path traversal
                    member_path = Path(member)
                    if ".." in member_path.parts or member.startswith("/") or member.startswith("\\"):
                        continue
                    if member.endswith("/") or member.endswith("\\") or member.startswith("__MACOSX"):
                        continue

                    sub_filename = member_path.name
                    if not sub_filename or sub_filename.startswith("."):
                        continue

                    # Attempt to resolve loader for member
                    try:
                        loader = self.loader_factory_fn(sub_filename)
                    except ValueError:
                        # Skip unsupported files within zip silently or collect names
                        continue

                    with archive.open(member) as member_file:
                        member_bytes = member_file.read()
                        member_stream = io.BytesIO(member_bytes)
                        # Tag source with archive hierarchy
                        full_source_tag = f"{filename}/{member}"
                        docs = loader.load(member_stream, full_source_tag)
                        documents.extend(docs)
                        extracted_names.append(full_source_tag)

                if not documents:
                    raise ValueError(
                        f"ZIP archive '{filename}' contained no supported or non-empty documents."
                    )

                return documents
        finally:
            if close_stream:
                zip_stream.close()
