"""Compatibility namespace kept empty after the full local split."""
"""Explicit compatibility boundaries for code that has not been ported yet."""

from .backends import JIMU_ROOT, LEGO_ROOT, REPO_ROOT, jimu_script, lego_command_root

__all__ = ["JIMU_ROOT", "LEGO_ROOT", "REPO_ROOT", "jimu_script", "lego_command_root"]
