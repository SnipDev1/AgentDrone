"""Unit tests for UI parcel upload and delete handlers.

Verifies:
- Upload handling (zip and raw files)
- Delete handling and cache synchronization
- Graceful handling of missing files / invalid parameters
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch
import zipfile

import streamlit as st

from src.dataset_manager import ParcelManager
from src.ui.app import handle_delete_plot, handle_upload_files, handle_upload_zip


def test_ui_handle_upload_files_validation(tmp_path: Path):
    """Verify handle_upload_files reports error when inputs are missing."""
    with patch("streamlit.error") as mock_err:
        handle_upload_files("", [])
        mock_err.assert_called_with("Пожалуйста, укажите кадастровый номер для создаваемой папки.")

    with patch("streamlit.error") as mock_err:
        handle_upload_files("5407055001999", [])
        mock_err.assert_called_with("Пожалуйста, выберите файлы съемки для загрузки.")


def test_ui_handle_upload_zip_and_delete(tmp_path: Path):
    """Verify handle_upload_zip deploys parcel and handle_delete_plot removes it."""
    plot_id = "5419130102999"
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr(f"{plot_id}/DJI_0001.JPG", b"sample_jpg_content")
        zf.writestr(f"{plot_id}/Timestamp.MRK", b"mrk_data")

    class MockUploadedFile:
        def getvalue(self):
            return zip_buffer.getvalue()

    with patch.object(ParcelManager, "get_dataset_root", return_value=tmp_path):
        # 1. Upload ZIP
        with patch("streamlit.rerun") as mock_rerun, patch("streamlit.toast"):
            handle_upload_zip(MockUploadedFile())
            assert mock_rerun.called
            assert (tmp_path / plot_id / "DJI_0001.JPG").exists()
            assert (
                st.session_state.get("pending_plot_id") == plot_id
                or st.session_state.get("selected_plot_id") == plot_id
            )

        # 2. Delete Plot
        with patch("streamlit.rerun") as mock_rerun, patch("streamlit.toast"):
            handle_delete_plot(plot_id)
            assert mock_rerun.called
            assert not (tmp_path / plot_id).exists()
            assert (
                st.session_state.get("pending_plot_id") is None
                or "selected_plot_id" not in st.session_state
            )
