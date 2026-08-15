"""
test_dataset.py — Unit tests for slicing and binary label derivation.

Tests:
  1. slice_volume returns correct shapes for all 3 views.
  2. slice_segmentation returns correct shapes.
  3. derive_binary_targets correctly extracts ET/TC/WT masks.
  4. All label values {0,1,2,3} are preserved through slicing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from spiking_useg.data.slicing import (
    slice_volume,
    slice_segmentation,
    derive_binary_targets,
    VIEW_CONFIG,
)


class TestSlicing:
    """Tests for volume → slice-sequence conversion."""

    CROP_SHAPE = (160, 192, 152)  # (H, W, D)

    def _make_data(self) -> np.ndarray:
        """Create a synthetic (4, 160, 192, 152) volume."""
        return np.random.rand(4, *self.CROP_SHAPE).astype(np.float32)

    def _make_seg(self) -> np.ndarray:
        """Create a synthetic (160, 192, 152) segmentation map with labels 0-3."""
        seg = np.zeros(self.CROP_SHAPE, dtype=np.int32)
        seg[40:80, 80:120, 50:100] = 1   # NCR
        seg[60:100, 90:130, 60:110] = 2  # ED
        seg[50:70, 95:115, 65:90] = 3    # ET
        return seg

    @pytest.mark.parametrize("view,axis,T,sH,sW", [
        ("sagittal", 0, 160, 192, 152),
        ("coronal",  1, 192, 160, 152),
        ("axial",    2, 152, 160, 192),
    ])
    def test_slice_volume_shape(self, view, axis, T, sH, sW):
        """slice_volume should return (T, 4, sH, sW)."""
        data = self._make_data()
        result = slice_volume(data, view)
        assert result.shape == (T, 4, sH, sW), (
            f"View={view}: expected ({T}, 4, {sH}, {sW}), got {result.shape}"
        )

    @pytest.mark.parametrize("view,axis,T,sH,sW", [
        ("sagittal", 0, 160, 192, 152),
        ("coronal",  1, 192, 160, 152),
        ("axial",    2, 152, 160, 192),
    ])
    def test_slice_segmentation_shape(self, view, axis, T, sH, sW):
        """slice_segmentation should return (T, sH, sW)."""
        seg = self._make_seg()
        result = slice_segmentation(seg, view)
        assert result.shape == (T, sH, sW), (
            f"View={view}: expected ({T}, {sH}, {sW}), got {result.shape}"
        )

    def test_slice_dtype(self):
        """slice_volume output should be float32."""
        data = self._make_data()
        result = slice_volume(data, "axial")
        assert result.dtype == np.float32

    def test_label_values_preserved(self):
        """All BraTS label values 0-3 should be present in sliced segmentation."""
        seg = self._make_seg()
        for view in ["axial", "coronal", "sagittal"]:
            sliced = slice_segmentation(seg, view)
            unique = set(np.unique(sliced).tolist())
            assert {0, 1, 2, 3}.issubset(unique), (
                f"View={view}: missing label values; got {unique}"
            )

    def test_unknown_view_raises(self):
        """Passing an invalid view name should raise ValueError."""
        data = self._make_data()
        with pytest.raises(ValueError, match="Unknown view"):
            slice_volume(data, "oblique")


class TestBinaryTargets:
    """Tests for derive_binary_targets (ET / TC / WT derivation)."""

    def test_et_is_label_3_only(self):
        """ET mask should be 1 only where label == 3."""
        seg = np.array([[0, 1, 2, 3, 1, 3, 0]], dtype=np.int32)
        targets = derive_binary_targets(seg)  # (3, 1, 7)
        et = targets[0]
        expected_et = (seg == 3).astype(np.float32)
        np.testing.assert_array_equal(et, expected_et)

    def test_tc_is_labels_1_and_3(self):
        """TC mask should be 1 where label ∈ {1, 3}."""
        seg = np.array([[0, 1, 2, 3, 0]], dtype=np.int32)
        targets = derive_binary_targets(seg)  # (3, 1, 5)
        tc = targets[1]
        expected_tc = ((seg == 1) | (seg == 3)).astype(np.float32)
        np.testing.assert_array_equal(tc, expected_tc)

    def test_wt_is_all_nonzero(self):
        """WT mask should be 1 wherever any tumor label is present."""
        seg = np.array([[0, 1, 2, 3, 0, 2]], dtype=np.int32)
        targets = derive_binary_targets(seg)  # (3, 1, 6)
        wt = targets[2]
        expected_wt = (seg > 0).astype(np.float32)
        np.testing.assert_array_equal(wt, expected_wt)

    def test_output_shape(self):
        """Output should be (3, H, W)."""
        seg = np.zeros((64, 64), dtype=np.int32)
        targets = derive_binary_targets(seg)
        assert targets.shape == (3, 64, 64)

    def test_hierarchical_inclusion(self):
        """ET ⊆ TC ⊆ WT voxelwise."""
        seg = np.array([[0, 1, 2, 3, 1, 3, 2, 0]], dtype=np.int32)
        targets = derive_binary_targets(seg)
        et, tc, wt = targets[0], targets[1], targets[2]
        # Every ET voxel must be in TC and WT
        assert np.all(et <= tc), "ET is not a subset of TC"
        assert np.all(tc <= wt), "TC is not a subset of WT"
