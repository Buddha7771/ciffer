from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .category import Category

if TYPE_CHECKING:
    import gemmi


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
    for key, category, item in [
        ("id", "_entry.", "id"),
        ("title", "_struct.", "title"),
        ("keywords", "_struct_keywords.", "pdbx_keywords"),
        ("deposit_date", "_pdbx_database_status.", "recvd_initial_deposition_date"),
    ]:
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
