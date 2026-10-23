"""
Text and Markdown Document Loader with encoding auto-detection.
"""

import hashlib
from pathlib import Path
from typing import BinaryIO, List, Union
from rag_core.models import Document, DocumentMetadata


class TextLoader:
    """Loads plain text (.txt) and markdown (.md) documents."""

    ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")

    def load(self, source: Union[str, Path, BinaryIO], filename: str) -> List[Document]:
        if isinstance(source, (str, Path)):
            with open(source, "rb") as f:
                content = f.read()
        elif hasattr(source, "read"):
            content = source.read()
            if hasattr(source, "seek"):
                source.seek(0)
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        if not content:
            raise ValueError(f"File '{filename}' is empty (0 bytes).")

        file_hash = hashlib.sha256(content).hexdigest()

        # Attempt decoding across standard text encodings
        text = None
        for enc in self.ENCODINGS:
            try:
                text = content.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise ValueError(f"Unable to decode text file '{filename}' with supported encodings.")

        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError(f"File '{filename}' contains only whitespace.")

        file_type = "text/markdown" if filename.lower().endswith(".md") else "text/plain"

        metadata = DocumentMetadata(
            source=filename,
            page=1,
            file_type=file_type,
            file_size_bytes=len(content),
            file_hash=file_hash,
        )

        return [Document(text=cleaned_text, metadata=metadata)]
