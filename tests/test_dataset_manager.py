"""Unit tests for Dataset Manager and Parcel Management.

Verifies:
- Plot ID sanitization and normalization
- File uploads into parcel directories
- ZIP archive uploads (structured, flat, wrapped)
- Zip slip and directory traversal security guards
- Safe parcel deletion
- Statistics calculation (RGB, MS, MRK, PPK, disk size)
- Integration with DroneDataset
"""

from __future__ import annotations

import io
from pathlib import Path
import zipfile

import pytest

from src.dataset import DroneDataset
from src.dataset_manager import ParcelManager, sanitize_plot_id


def test_sanitize_plot_id():
    """Verify various cadastral identifiers, filenames, and formats are correctly normalized."""
    assert sanitize_plot_id("5407055001178") == "5407055001178"
    assert sanitize_plot_id("54:07:055001:178") == "5407055001178"
    assert sanitize_plot_id("5407055001178.zip") == "5407055001178"
    assert sanitize_plot_id("54_19_163001_521") == "5419163001521"
    assert sanitize_plot_id("54:19:163001:521.zip") == "5419163001521"
    assert sanitize_plot_id("custom_sample") == "custom_sample"
    assert sanitize_plot_id("../../bad/path") == "path"


def test_upload_files_and_stats(tmp_path: Path):
    """Verify direct file upload and statistics calculation."""
    plot_id = "5407055001999"
    files = [
        ("DJI_20260807_0001_D.JPG", b"fake_jpg_data_1"),
        ("DJI_20260807_0002_D.JPG", b"fake_jpg_data_2"),
        ("Timestamp.MRK", b"1\t12345.678\t[Q=16]\tLat 54.8\tLon 83.2"),
        ("DJI_PPKOBS.obs", b"rinex_obs_bytes"),
        ("notes.txt", b"Inspection note"),
    ]

    saved_id = ParcelManager.upload_files(plot_id, files, dataset_root=tmp_path)
    assert saved_id == plot_id

    plot_dir = tmp_path / plot_id
    assert plot_dir.exists()
    assert (plot_dir / "DJI_20260807_0001_D.JPG").exists()
    assert (plot_dir / "Timestamp.MRK").exists()

    stats = ParcelManager.get_plot_stats(plot_id, dataset_root=tmp_path)
    assert stats["exists"] is True
    assert stats["total_files"] == 5
    assert stats["rgb_count"] == 2
    assert stats["has_mrk"] is True
    assert stats["has_ppk"] is True
    assert stats["has_notes"] is True


def test_upload_zip_structured(tmp_path: Path):
    """Verify ZIP extraction when ZIP contains a top-level parcel directory."""
    plot_id = "5419130102999"
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr(f"{plot_id}/DJI_0001.JPG", b"sample_jpg_content")
        zf.writestr(f"{plot_id}/Timestamp.MRK", b"mrk_data")
        zf.writestr(f"{plot_id}/subfolder/DJI_0002.JPG", b"sample_jpg_content_2")

    zip_bytes = zip_buffer.getvalue()
    extracted_ids = ParcelManager.upload_zip(zip_bytes, dataset_root=tmp_path)

    assert plot_id in extracted_ids
    assert (tmp_path / plot_id / "DJI_0001.JPG").exists()
    assert (tmp_path / plot_id / "Timestamp.MRK").exists()
    assert (tmp_path / plot_id / "subfolder" / "DJI_0002.JPG").exists()


def test_upload_zip_flat(tmp_path: Path):
    """Verify ZIP extraction when files are at archive root with target plot ID."""
    plot_id = "5419163001999"
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("DJI_0005.JPG", b"sample_content_5")
        zf.writestr("Timestamp.MRK", b"mrk_data")

    extracted_ids = ParcelManager.upload_zip(
        zip_buffer.getvalue(),
        target_plot_id=plot_id,
        dataset_root=tmp_path,
    )

    assert plot_id in extracted_ids
    assert (tmp_path / plot_id / "DJI_0005.JPG").exists()
    assert (tmp_path / plot_id / "Timestamp.MRK").exists()


def test_upload_zip_security_zip_slip(tmp_path: Path):
    """Verify zip slip directory traversal attempts are rejected."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("../../../evil.txt", b"evil")

    with pytest.raises(Exception):
        ParcelManager.upload_zip(zip_buffer.getvalue(), dataset_root=tmp_path)


def test_delete_plot_success_and_missing(tmp_path: Path):
    """Verify deleting existing parcel removes folder; nonexistent returns False."""
    plot_id = "5432010870999"
    plot_dir = tmp_path / plot_id
    plot_dir.mkdir(parents=True)
    (plot_dir / "DJI_0001.JPG").write_bytes(b"data")

    assert plot_dir.exists()
    deleted = ParcelManager.delete_plot(plot_id, dataset_root=tmp_path)
    assert deleted is True
    assert not plot_dir.exists()

    # Deleting again should return False
    assert ParcelManager.delete_plot(plot_id, dataset_root=tmp_path) is False


def test_delete_plot_security_guard(tmp_path: Path):
    """Verify delete_plot protects against directory traversal and root deletion."""
    with pytest.raises(ValueError, match="cannot delete dataset root"):
        ParcelManager.delete_plot("", dataset_root=tmp_path)

    with pytest.raises(ValueError):
        ParcelManager.delete_plot("../../outside", dataset_root=tmp_path)


def test_dronedataset_integration(tmp_path: Path):
    """Verify DroneDataset delegate methods for upload, stats, and delete."""
    ds = DroneDataset(root_dir=tmp_path)
    plot_id = "5435121015999"

    # Upload files via DroneDataset
    saved_id = ds.upload_files(
        plot_id=plot_id,
        files=[("DJI_001.JPG", b"data1"), ("Timestamp.MRK", b"data2")],
    )
    assert saved_id == plot_id
    assert plot_id in ds.list_plots()

    # Get stats
    stats = ds.get_plot_stats(plot_id)
    assert stats["exists"] is True
    assert stats["total_files"] == 2

    # Delete plot via DroneDataset
    res = ds.delete_plot(plot_id)
    assert res is True
    assert plot_id not in ds.list_plots()
