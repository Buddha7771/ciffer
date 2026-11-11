from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .category import Category, to_cif

if TYPE_CHECKING:
    from collections.abc import Mapping

    import gemmi

    from .hierarchy import Items

# The keys entry_metadata reads, which metadata_categories writes back.
ENTRY_KEYS = {
    "id",
    "title",
    "keywords",
    "deposit_date",
    "methods",
    "authors",
    "resolution",
    "r_free",
    "release_date",
    "doi",
}
# Metadata keys held in one text item, with its category and item name.
ENTRY_TEXT = [
    ("title", "_struct.", "title"),
    ("keywords", "_struct_keywords.", "pdbx_keywords"),
    ("deposit_date", "_pdbx_database_status.", "recvd_initial_deposition_date"),
]


def entry_metadata(block: gemmi.cif.Block) -> dict[str, Any]:
    """Return the entry values present in the file.

    The resolution comes from ``_refine``, else ``_em_3d_reconstruction``, else
    ``_reflns``. The release date is the first revision date.

    Examples
    --------
    .. code-block:: python

        entry_metadata(block)
        # {'id': '1EJG', 'title': 'CRAMBIN AT ULTRA-HIGH RESOLUTION: ...',
        #  'methods': ['X-RAY DIFFRACTION'], 'resolution': 0.54, 'r_free': 0.094,
        #  'release_date': '2000-04-05', 'doi': '10.1073/pnas.97.7.3171', ...}

    """

    def _values(category: str, item: str) -> list[str]:
        return [v for v in Category(block, category).get(item).tolist() if v]

    metadata: dict[str, Any] = {}
    for key, category, item in [("id", "_entry.", "id"), *ENTRY_TEXT]:
        if values := _values(category, item):
            metadata[key] = values[0]

    for key, category, item in [
        ("methods", "_exptl.", "method"),
        ("authors", "_audit_author.", "name"),
    ]:
        if values := _values(category, item):
            metadata[key] = values

    resolution = (
        _values("_refine.", "ls_d_res_high")
        or _values("_em_3d_reconstruction.", "resolution")
        or _values("_reflns.", "d_resolution_high")
    )
    if resolution:
        metadata["resolution"] = float(resolution[0])
    if r_free := _values("_refine.", "ls_R_factor_R_free"):
        metadata["r_free"] = float(r_free[0])

    if revisions := _values("_pdbx_audit_revision_history.", "revision_date"):
        metadata["release_date"] = min(revisions)

    citation = Category(block, "_citation.")
    primary = citation.get("id") == "primary"
    if doi := [v for v in citation.get("pdbx_database_id_DOI")[primary] if v]:
        metadata["doi"] = str(doi[0])
    return metadata


def metadata_categories(metadata: Mapping[str, Any]) -> dict[str, Items]:
    """Return the categories that hold the entry values ``entry_metadata`` reads.

    The release date is written as the only revision.

    Raises
    ------
    TypeError
        If ``methods`` or ``authors`` is a string rather than a list.
    """
    # A key set to None is absent.
    metadata = {key: value for key, value in metadata.items() if value is not None}
    for key in ("methods", "authors"):
        if isinstance(metadata.get(key), str):
            msg = f"metadata[{key!r}] must be a list, such as [{metadata[key]!r}]."
            raise TypeError(msg)
    entry_id = to_cif([metadata.get("id", "")])
    methods = [str(method) for method in metadata.get("methods", [])]

    categories: dict[str, Items] = {}
    if "id" in metadata:
        categories["_entry."] = {"id": entry_id}
    for key, category, item in ENTRY_TEXT:
        if key in metadata:
            categories[category] = {"entry_id": entry_id, item: to_cif([metadata[key]])}
    if methods:
        categories["_exptl."] = {
            "entry_id": entry_id * len(methods),
            "method": to_cif(methods),
        }
    if "authors" in metadata:
        authors = to_cif(metadata["authors"])
        categories["_audit_author."] = {
            "name": authors,
            "pdbx_ordinal": to_cif(range(1, len(authors) + 1)),
        }

    categories |= _refinement_categories(metadata, entry_id, methods)
    if "release_date" in metadata:
        categories["_pdbx_audit_revision_history."] = {
            "ordinal": ["1"],
            "data_content_type": ["Structure model"],
            "major_revision": ["1"],
            "minor_revision": ["0"],
            "revision_date": to_cif([metadata["release_date"]]),
        }
    if "doi" in metadata:
        categories["_citation."] = {
            "id": ["primary"],
            "pdbx_database_id_DOI": to_cif([metadata["doi"]]),
        }
    return categories


def _refinement_categories(
    metadata: Mapping[str, Any], entry_id: list[str | bool | None], methods: list[str]
) -> dict[str, Items]:
    """Return ``_refine``, with an EM resolution in ``_em_3d_reconstruction``."""
    categories: dict[str, Items] = {}
    refine: Items = {}
    if "resolution" in metadata:
        resolution = to_cif([metadata["resolution"]])
        if "ELECTRON MICROSCOPY" in methods:
            categories["_em_3d_reconstruction."] = {
                "id": ["1"],
                "resolution": resolution,
            }
        else:
            refine["ls_d_res_high"] = resolution
    if "r_free" in metadata:
        refine["ls_R_factor_R_free"] = to_cif([metadata["r_free"]])
    if refine:
        # gemmi reads no resolution from _refine without pdbx_refine_id.
        refine_id = to_cif(methods[:1] or [""])
        categories["_refine."] = {
            "entry_id": entry_id,
            "pdbx_refine_id": refine_id,
            **refine,
        }
    return categories
