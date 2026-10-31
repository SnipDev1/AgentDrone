"""Drone Dataset loader, inspector, and validator.

Provides a unified interface for working with DJI drone aerial imagery
and PPK telemetry data across team environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import sys
from typing import BinaryIO

from .config import find_dataset_root
from .dataset_manager import ParcelManager

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


@dataclass
class PlotData:
    """Represents data collected for a single cadastral plot."""
    plot_id: str
    dir_path: Path
    rgb_images: list[Path] = field(default_factory=list)
    multispectral_images: dict[str, list[Path]] = field(default_factory=lambda: {
        'NIR': [], 'RE': [], 'R': [], 'G': []
    })
    mrk_files: list[Path] = field(default_factory=list)
    ppk_obs: list[Path] = field(default_factory=list)
    ppk_nav: list[Path] = field(default_factory=list)
    ppk_raw: list[Path] = field(default_factory=list)
    notes: list[Path] = field(default_factory=list)
    other_files: list[Path] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        ms_count = sum(len(v) for v in self.multispectral_images.values())
        return (
            len(self.rgb_images) +
            ms_count +
            len(self.mrk_files) +
            len(self.ppk_obs) +
            len(self.ppk_nav) +
            len(self.ppk_raw) +
            len(self.notes) +
            len(self.other_files)
        )

    @property
    def total_size_bytes(self) -> int:
        all_files = (
            self.rgb_images +
            [f for band_files in self.multispectral_images.values() for f in band_files] +
            self.mrk_files +
            self.ppk_obs +
            self.ppk_nav +
            self.ppk_raw +
            self.notes +
            self.other_files
        )
        return sum(f.stat().st_size for f in all_files if f.is_file())


class DroneDataset:
    """Unified dataset manager for AgentDrone."""

    def __init__(self, root_dir: str | Path | None = None):
        self.root_dir = find_dataset_root(root_dir)

    def is_available(self) -> bool:
        """Check if dataset root exists and contains cadastral folders."""
        if not self.root_dir.exists() or not self.root_dir.is_dir():
            return False
        plots = self.list_plots()
        return len(plots) > 0

    def list_plots(self) -> list[str]:
        """List all cadastral plot IDs found in the dataset."""
        if not self.root_dir.exists():
            return []
        plots = []
        for d in self.root_dir.iterdir():
            if d.is_dir() and not d.name.startswith('.'):
                clean_digits = re.sub(r"[^\d]", "", d.name)
                if (d.name.isdigit() and len(d.name) >= 10) or (len(clean_digits) >= 10) or any(d.glob("*.JPG")):
                    plots.append(d.name)
        return sorted(plots)

    def get_plot(self, plot_id: str) -> PlotData:
        """Retrieve all categorized files for a specific cadastral plot."""
        plot_dir = self.root_dir / plot_id
        if not plot_dir.exists() or not plot_dir.is_dir():
            raise FileNotFoundError(f"Plot directory not found: {plot_dir}")

        plot = PlotData(plot_id=plot_id, dir_path=plot_dir)

        # Walk through plot directory (including any subflights)
        for item in sorted(plot_dir.rglob('*')):
            if not item.is_file():
                continue

            name = item.name
            ext = item.suffix.lower()

            if ext in ('.jpg', '.jpeg'):
                plot.rgb_images.append(item)
            elif ext in ('.tif', '.tiff'):
                # Multispectral bands: MS_NIR, MS_RE, MS_R, MS_G
                upper_name = name.upper()
                if 'MS_NIR' in upper_name:
                    plot.multispectral_images['NIR'].append(item)
                elif 'MS_RE' in upper_name:
                    plot.multispectral_images['RE'].append(item)
                elif 'MS_R' in upper_name:
                    plot.multispectral_images['R'].append(item)
                elif 'MS_G' in upper_name:
                    plot.multispectral_images['G'].append(item)
                else:
                    plot.other_files.append(item)
            elif ext == '.mrk':
                plot.mrk_files.append(item)
            elif ext == '.obs':
                plot.ppk_obs.append(item)
            elif ext == '.nav':
                plot.ppk_nav.append(item)
            elif ext == '.bin':
                plot.ppk_raw.append(item)
            elif ext == '.txt':
                plot.notes.append(item)
            else:
                plot.other_files.append(item)

        return plot

    def get_all_plots(self) -> dict[str, PlotData]:
        """Retrieve data for all plots as a dictionary mapping plot_id -> PlotData."""
        return {pid: self.get_plot(pid) for pid in self.list_plots()}

    def delete_plot(self, plot_id: str) -> bool:
        """Safely delete a parcel folder from this dataset root."""
        return ParcelManager.delete_plot(plot_id, dataset_root=self.root_dir)

    def upload_zip(
        self,
        zip_source: bytes | BinaryIO | Path | str,
        target_plot_id: str | None = None,
    ) -> list[str]:
        """Extract a parcel archive (.zip) into this dataset root."""
        return ParcelManager.upload_zip(zip_source, target_plot_id=target_plot_id, dataset_root=self.root_dir)

    def upload_files(
        self,
        plot_id: str,
        files: list[tuple[str, bytes]],
    ) -> str:
        """Save a list of (filename, file_bytes) into this dataset root under plot_id."""
        return ParcelManager.upload_files(plot_id, files, dataset_root=self.root_dir)

    def get_plot_stats(self, plot_id: str) -> dict:
        """Retrieve file metrics and storage stats for a parcel folder."""
        return ParcelManager.get_plot_stats(plot_id, dataset_root=self.root_dir)

    def validate(self) -> dict:
        """Perform validation and return comprehensive report."""
        if not self.is_available():
            return {
                'status': 'error',
                'message': f"Dataset directory not found or empty: {self.root_dir}",
                'plots_count': 0,
                'total_files': 0,
                'total_size_mb': 0.0,
            }

        plots = self.get_all_plots()
        total_files = sum(p.total_files for p in plots.values())
        total_bytes = sum(p.total_size_bytes for p in plots.values())
        total_rgb = sum(len(p.rgb_images) for p in plots.values())
        total_ms = sum(sum(len(v) for v in p.multispectral_images.values()) for p in plots.values())
        total_mrk = sum(len(p.mrk_files) for p in plots.values())
        total_ppk = sum(len(p.ppk_obs) + len(p.ppk_nav) + len(p.ppk_raw) for p in plots.values())

        issues = []
        for pid, p in plots.items():
            if not p.rgb_images and not p.multispectral_images:
                issues.append(f"Plot {pid}: No imagery found.")
            if not p.mrk_files:
                issues.append(f"Plot {pid}: Missing .MRK timestamp file.")

        return {
            'status': 'ok' if not issues else 'warning',
            'root_dir': str(self.root_dir),
            'plots_count': len(plots),
            'total_files': total_files,
            'total_size_mb': total_bytes / (1024 * 1024),
            'rgb_images': total_rgb,
            'multispectral_images': total_ms,
            'mrk_files': total_mrk,
            'ppk_files': total_ppk,
            'issues': issues,
        }

    def print_summary(self):
        """Print a human-readable dataset summary and integrity check."""
        print("=" * 60)
        print("AgentDrone Dataset Summary")
        print("=" * 60)
        print(f"Dataset root: {self.root_dir}")

        if not self.is_available():
            print("\n[ERROR] Dataset not found or contains no valid cadastral plots!")
            print(f"Expected location: {self.root_dir}")
            print("\nHow to fix:")
            print("  1. Place the dataset inside 'dataset/' or configure DATASET_DIR in .env")
            print("  2. Or run: python download_disk.py")
            print("=" * 60)
            return

        report = self.validate()
        print(f"Status:               {report['status'].upper()}")
        print(f"Cadastral plots:      {report['plots_count']}")
        print(f"Total files:          {report['total_files']}")
        print(f"Total size:           {report['total_size_mb']:.2f} MB")
        print(f"RGB photos (.JPG):    {report['rgb_images']}")
        print(f"Multispectral (.TIF): {report['multispectral_images']}")
        print(f"MRK timestamps:       {report['mrk_files']}")
        print(f"PPK logs (.obs/.nav): {report['ppk_files']}")

        print("\nCadastral Plots Breakdown:")
        print(f"{'Plot ID':<18} {'RGB':<6} {'MS':<6} {'MRK':<6} {'PPK':<6} {'Size (MB)':<10}")
        print("-" * 56)
        for pid, p in self.get_all_plots().items():
            ms_count = sum(len(v) for v in p.multispectral_images.values())
            ppk_count = len(p.ppk_obs) + len(p.ppk_nav) + len(p.ppk_raw)
            size_mb = p.total_size_bytes / (1024 * 1024)
            print(
                f"{pid:<18} {len(p.rgb_images):<6} {ms_count:<6} "
                f"{len(p.mrk_files):<6} {ppk_count:<6} {size_mb:>8.2f}"
            )

        if report['issues']:
            print("\nWarnings:")
            for issue in report['issues']:
                print(f"  - {issue}")
        else:
            print("\n[OK] All plots validated successfully!")
        print("=" * 60)


if __name__ == '__main__':
    ds = DroneDataset()
    ds.print_summary()
