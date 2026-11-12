from __future__ import annotations

import gzip
import urllib.error
import urllib.request
from http import HTTPStatus


def download_mmcif(pdb_id: str) -> bytes:
    """Return the uncompressed mmCIF file of an entry, downloaded from RCSB.

    Raises
    ------
    FileNotFoundError
        If RCSB has no entry of the id.

    """
    url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
    request = urllib.request.Request(url, headers={"User-Agent": "ciffer"})
    try:
        # Only the id varies in the URL, and fetch checks it.
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return gzip.decompress(response.read())
    except urllib.error.HTTPError as error:
        if error.code == HTTPStatus.NOT_FOUND:
            msg = f"No entry {pdb_id!r} at RCSB."
            raise FileNotFoundError(msg) from error
        raise
