from __future__ import annotations

import io
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
from zstandard import ZstdCompressor, ZstdDecompressor

ARRAY_KEY = "__array__"
HEADER_SIZE_BYTES = 8


def dump_bytes(data: Mapping[str, Any]) -> bytes:
    """Serialize a dictionary containing numpy arrays to zstd-compressed bytes.

    Parameters
    ----------
    data: Mapping[str, Any]
        Nested dicts with string keys, lists, numpy arrays, numpy scalars, and
        JSON values.

    Raises
    ------
    TypeError
        If a value cannot be saved. The message names where it was found.

    Notes
    -----
    Numpy scalars come back as Python numbers, and tuples as lists.
    """
    blobs: list[bytes] = []

    def _encode(value: Any, path: str) -> Any:
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                msg = f"Cannot save {path}: object arrays are not supported."
                raise TypeError(msg)
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            blobs.append(buffer.getvalue())
            return {ARRAY_KEY: len(blobs) - 1}

        if isinstance(value, np.generic):
            return _encode(value.item(), path)

        if isinstance(value, Mapping):
            encoded = {}
            for key, item in value.items():
                if not isinstance(key, str) or key == ARRAY_KEY:
                    msg = f"Cannot save {path}: {key!r} cannot be a key."
                    raise TypeError(msg)
                encoded[key] = _encode(item, f"{path}.{key}" if path else key)
            return encoded

        if isinstance(value, list | tuple):
            return [_encode(item, f"{path}[{i}]") for i, item in enumerate(value)]

        if value is None or isinstance(value, str | int | float):
            return value

        msg = f"Cannot save {path}: {type(value).__name__} is not supported."
        raise TypeError(msg)

    header = {"template": _encode(data, ""), "sizes": [len(b) for b in blobs]}
    header_bytes = json.dumps(header).encode("utf-8")

    payload = b"".join(
        [len(header_bytes).to_bytes(HEADER_SIZE_BYTES, "little"), header_bytes, *blobs]
    )
    return ZstdCompressor().compress(payload)


def load_bytes(byte_data: bytes) -> dict[str, Any]:
    """Deserialize zstd-compressed bytes back into a dictionary."""
    payload = memoryview(ZstdDecompressor().decompress(byte_data))

    header_end = HEADER_SIZE_BYTES + int.from_bytes(
        payload[:HEADER_SIZE_BYTES], "little"
    )
    header = json.loads(bytes(payload[HEADER_SIZE_BYTES:header_end]))

    arrays = []
    offset = header_end
    for size in header["sizes"]:
        buffer = io.BytesIO(payload[offset : offset + size])
        arrays.append(np.load(buffer, allow_pickle=False))
        offset += size

    def _decode(value: Any) -> Any:
        if isinstance(value, dict):
            if ARRAY_KEY in value:
                return arrays[value[ARRAY_KEY]]
            return {key: _decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_decode(item) for item in value]
        return value

    return _decode(header["template"])
