"""Dataset Manager for AgentDrone.

Provides robust, production-ready parcel folder operations:
- Uploading parcel folders via ZIP archives or direct file collections
- Deleting parcel folders with path traversal and integrity guards
- Sanitizing cadastral and parcel folder identifiers
- Computing parcel folder metrics (RGB images, multispectral, MRK, PPK, disk size)
"""

from __future__ import annotations

import io
from pathlib import Path
import re
import shutil
from typing import BinaryIO
import zipfile

from .config import find_dataset_root
from .ingestion.cadastral_registry import CadastralRegistry, normalize_cadastral_id


def is_cadastral_digits(name: str) -> bool:
    """Return True only if `name` looks like a Russian cadastral number (not a DJI YYYYMMDD timestamp)."""
    if not name or "DJI" in name.upper():
        return False
    digits = re.sub(r"[^\d]", "", name)
    if digits.startswith(("2020", "2021", "2022", "2023", "2024", "2025", "2026", "2027", "2028", "2029")):
        return False
    return 10 <= len(digits) <= 16


def sanitize_plot_id(name: str) -> str:
    """Sanitize and normalize a raw folder name, filename, or cadastral ID into a safe identifier.

    Examples:
        '54:07:055001:178' -> '5407055001178'
        '5407055001178.zip' -> '5407055001178'
        '54_19_163001_521' -> '5419163001521'
        'plot-123' -> 'plot_123'
    """
    clean_base = Path(name).stem if "." in name else name
    digits_only = re.sub(r"[^\d]", "", clean_base)
    if is_cadastral_digits(clean_base):
        return digits_only

    # Fallback to sanitized alphanumeric if not standard Russian cadastral digits
    safe_name = re.sub(r"[^\w\-]", "_", clean_base).strip("._")
    return safe_name or "uploaded_plot"


class ParcelManager:
    """Service for managing drone inspection folders on the server filesystem."""

    @classmethod
    def get_dataset_root(cls, custom_path: str | Path | None = None) -> Path:
        """Resolve dataset root directory."""
        if custom_path is not None:
            p = Path(custom_path)
            return p.resolve()
        return find_dataset_root().resolve()

    @classmethod
    def list_plots(cls, dataset_root: Path | None = None) -> list[str]:
        """List all valid parcel directories in the dataset root."""
        root = cls.get_dataset_root(dataset_root)
        if not root.exists() or not root.is_dir():
            return []

        plots = []
        for d in root.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                # Accepts numeric folders with length >= 10 or valid plot directories containing files
                if (d.name.isdigit() and len(d.name) >= 10) or any(d.iterdir()):
                    plots.append(d.name)
        return sorted(plots)

    @classmethod
    def get_plot_stats(cls, plot_id: str, dataset_root: Path | None = None) -> dict:
        """Retrieve file metrics and storage statistics for a parcel folder."""
        root = cls.get_dataset_root(dataset_root)
        clean_id = sanitize_plot_id(plot_id)
        plot_dir = root / clean_id
        if not plot_dir.exists() and (root / plot_id).exists():
            plot_dir = root / plot_id

        if not plot_dir.exists() or not plot_dir.is_dir():
            return {
                "plot_id": clean_id,
                "cadastral_id": normalize_cadastral_id(clean_id),
                "exists": False,
                "dir_path": str(plot_dir),
                "total_files": 0,
                "size_bytes": 0,
                "size_mb": 0.0,
                "rgb_count": 0,
                "ms_count": 0,
                "has_mrk": False,
                "has_ppk": False,
                "has_notes": False,
                "file_names": [],
            }

        all_files = [f for f in plot_dir.rglob("*") if f.is_file()]
        rgb_files = [f for f in all_files if f.suffix.lower() in (".jpg", ".jpeg")]
        ms_files = [f for f in all_files if f.suffix.lower() in (".tif", ".tiff")]
        mrk_files = [f for f in all_files if f.suffix.lower() == ".mrk"]
        ppk_files = [f for f in all_files if f.suffix.lower() in (".obs", ".nav", ".bin")]
        notes_files = [f for f in all_files if f.suffix.lower() == ".txt"]
        total_bytes = sum(f.stat().st_size for f in all_files)

        return {
            "plot_id": plot_dir.name,
            "cadastral_id": normalize_cadastral_id(plot_dir.name),
            "exists": True,
            "dir_path": str(plot_dir),
            "total_files": len(all_files),
            "size_bytes": total_bytes,
            "size_mb": round(total_bytes / (1024 * 1024), 2),
            "rgb_count": len(rgb_files),
            "ms_count": len(ms_files),
            "has_mrk": len(mrk_files) > 0,
            "has_ppk": len(ppk_files) > 0,
            "has_notes": len(notes_files) > 0,
            "file_names": [f.name for f in all_files],
        }

    @classmethod
    def delete_plot(cls, plot_id: str, dataset_root: Path | None = None) -> bool:
        """Safely delete a parcel folder from the dataset root.

        Guards against directory traversal and root deletion.
        """
        root = cls.get_dataset_root(dataset_root)
        if not root.exists():
            return False

        clean_id = sanitize_plot_id(plot_id)
        candidate_dirs = [root / clean_id, root / plot_id]

        target_dir = None
        for cand in candidate_dirs:
            resolved = cand.resolve()
            # Guard against directory traversal or root deletion
            try:
                resolved.relative_to(root)
            except ValueError:
                raise ValueError(f"Security error: target path {resolved} is outside root {root}")

            if resolved == root:
                raise ValueError("Security error: cannot delete dataset root directory")

            if resolved.exists() and resolved.is_dir():
                target_dir = resolved
                break

        if target_dir is None:
            return False

        shutil.rmtree(target_dir)
        return True

    @classmethod
    def delete_all_plots(cls, dataset_root: Path | None = None) -> int:
        """Delete all parcel folders from the dataset root and return deleted count."""
        plots = cls.list_plots(dataset_root=dataset_root)
        deleted_count = 0
        for pid in plots:
            if cls.delete_plot(pid, dataset_root=dataset_root):
                deleted_count += 1
        return deleted_count

    @classmethod
    def upload_files(
        cls,
        plot_id: str,
        files: list[tuple[str, bytes]],
        dataset_root: Path | None = None,
    ) -> str:
        """Save a list of (filename, bytes) into the specified parcel directory."""
        if not files:
            raise ValueError("No files provided for upload")

        clean_id = sanitize_plot_id(plot_id)
        root = cls.get_dataset_root(dataset_root)
        root.mkdir(parents=True, exist_ok=True)

        target_dir = (root / clean_id).resolve()
        try:
            target_dir.relative_to(root)
        except ValueError:
            raise ValueError(f"Security error: target path {target_dir} is outside root {root}")

        target_dir.mkdir(parents=True, exist_ok=True)

        for filename, data in files:
            # Strip path components to prevent directory traversal
            safe_name = Path(filename).name
            if not safe_name:
                continue
            dest_file = target_dir / safe_name
            with open(dest_file, "wb") as f:
                f.write(data)

        CadastralRegistry.get_parcel(clean_id, fetch_online=True, plot_dir=target_dir)
        return clean_id

    @classmethod
    def upload_zip(
        cls,
        zip_source: bytes | BinaryIO | Path | str,
        target_plot_id: str | None = None,
        dataset_root: Path | None = None,
    ) -> list[str]:
        """Extract a ZIP archive of one or more parcel folders into dataset root.

        Supports:
        1. Archives containing top-level parcel directories (e.g. '5407055001178/DJI_...')
        2. Flat archives with direct files (using target_plot_id or zip name)
        3. Wrapped archives (e.g. 'flight_data/5407055001178/...')

        Includes strict Zip Slip directory traversal prevention.
        """
        root = cls.get_dataset_root(dataset_root)
        root.mkdir(parents=True, exist_ok=True)

        if isinstance(zip_source, (bytes, bytearray)):
            zip_buffer: BinaryIO = io.BytesIO(zip_source)
        elif isinstance(zip_source, (str, Path)):
            zip_buffer = open(zip_source, "rb")
        else:
            zip_buffer = zip_source

        extracted_plot_ids: set[str] = set()

        with zipfile.ZipFile(zip_buffer, "r") as zf:
            members = [m for m in zf.infolist() if not m.is_dir() and not m.filename.startswith("__MACOSX")]
            if not members:
                raise ValueError("ZIP archive contains no valid files")

            # Validate against Zip Slip traversal
            for m in members:
                parts = Path(m.filename).parts
                if ".." in parts or any(p.startswith("/") or p.startswith("\\") for p in parts):
                    raise ValueError(f"Zip slip security error: invalid path in archive '{m.filename}'")

            # Analyze archive structure: inspect non-empty path parts
            first_parts = {
                Path(m.filename).parts[0]
                for m in members
                if len(Path(m.filename).parts) > 1
            }

            # Check if top-level directory names represent parcel IDs (not DJI_2026... folders)
            cadastral_dirs = [
                d for d in first_parts
                if is_cadastral_digits(d)
            ]

            if cadastral_dirs and not target_plot_id:
                # Structure A: ZIP contains top-level parcel folders
                for member in members:
                    parts = Path(member.filename).parts
                    top_dir = parts[0]
                    clean_id = sanitize_plot_id(top_dir)
                    extracted_plot_ids.add(clean_id)

                    # Relative path inside the plot folder
                    sub_path = Path(*parts[1:])
                    dest_path = (root / clean_id / sub_path).resolve()

                    # Zip slip check
                    dest_path.relative_to(root)
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                    with zf.open(member) as src, open(dest_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)

            else:
                # Check for second-level parcel folders (e.g. 'root_folder/5407055001178/...')
                second_level_dirs = set()
                for m in members:
                    parts = Path(m.filename).parts
                    if len(parts) >= 3 and is_cadastral_digits(parts[1]):
                        second_level_dirs.add(parts[1])

                if second_level_dirs and not target_plot_id:
                    # Structure C: Wrapped inside a single parent folder
                    for member in members:
                        parts = Path(member.filename).parts
                        if len(parts) >= 2 and is_cadastral_digits(parts[1]):
                            clean_id = sanitize_plot_id(parts[1])
                            extracted_plot_ids.add(clean_id)
                            sub_path = Path(*parts[2:])
                            dest_path = (root / clean_id / sub_path).resolve()
                            dest_path.relative_to(root)
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            with zf.open(member) as src, open(dest_path, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                else:
                    # Structure B: Flat files or DJI flight subfolders inside zip
                    dest_id = target_plot_id
                    if not dest_id and hasattr(zip_source, "name") and zip_source.name:
                        dest_id = sanitize_plot_id(zip_source.name)
                    if not dest_id and isinstance(zip_source, (str, Path)):
                        dest_id = sanitize_plot_id(str(zip_source))
                    if not dest_id:
                        dest_id = "uploaded_plot"

                    clean_id = sanitize_plot_id(dest_id)
                    extracted_plot_ids.add(clean_id)
                    plot_dir = root / clean_id
                    plot_dir.mkdir(parents=True, exist_ok=True)

                    for member in members:
                        safe_filename = Path(member.filename).name
                        dest_path = (plot_dir / safe_filename).resolve()
                        dest_path.relative_to(root)
                        with zf.open(member) as src, open(dest_path, "wb") as dst:
                            shutil.copyfileobj(src, dst)

        final_ids: list[str] = []
        for pid in sorted(extracted_plot_ids):
            plot_dir = root / pid
            parcel = CadastralRegistry.get_parcel(pid, fetch_online=True, plot_dir=plot_dir)
            if (
                parcel is not None
                and not is_cadastral_digits(pid)
                and is_cadastral_digits(parcel.raw_folder_name)
            ):
                new_dir = root / parcel.raw_folder_name
                if new_dir.exists() and new_dir != plot_dir:
                    shutil.rmtree(new_dir)
                if plot_dir.exists() and new_dir != plot_dir:
                    plot_dir.rename(new_dir)
                final_ids.append(parcel.raw_folder_name)
            else:
                final_ids.append(pid)

        return sorted(set(final_ids))
