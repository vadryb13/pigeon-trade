"""Тесты edge-cases для CPCV (CombinatorialPurgedCV) и purged_kfold_indices.

Покрывают ветки, которые существуют в коде но не покрыты тестами:
- ValueError при n_test_splits >= n_splits
- n_paths
- _get_test_ranges с разными параметрами
- Purge logic
- Embargo при маленьких данных
"""
from __future__ import annotations

import pandas as pd
import pytest

from aqr.validation.cpcv import (
    CombinatorialPurgedCV,
    _get_test_ranges,
    purged_kfold_indices,
)


class TestGetTestRanges:
    def test_not_enough_samples_raises(self):
        """Если данных меньше чем n_splits — ValueError."""
        ts = pd.date_range("2024-01-01", periods=3, freq="D")
        with pytest.raises(ValueError, match="Not enough samples"):
            _get_test_ranges(ts, n_splits=6, n_test_splits=2)

    def test_default_n_test_splits_2(self):
        ts = pd.date_range("2024-01-01", periods=600, freq="D")
        ranges = _get_test_ranges(ts, n_splits=6, n_test_splits=2)
        # C(6, 2) = 15 путей
        assert len(ranges) == 15
        # Каждый путь = 2 фолда
        for path in ranges:
            assert len(path) == 2

    def test_single_test_split(self):
        """n_test_splits=1 → один фолд на путь."""
        ts = pd.date_range("2024-01-01", periods=600, freq="D")
        ranges = _get_test_ranges(ts, n_splits=6, n_test_splits=1)
        # C(6, 1) = 6 путей
        assert len(ranges) == 6


class TestCombinatorialPurgedCVInit:
    def test_n_test_splits_must_be_less_than_n_splits(self):
        """Если n_test_splits >= n_splits — ValueError."""
        with pytest.raises(ValueError, match="n_test_splits must be < n_splits"):
            CombinatorialPurgedCV(n_splits=3, n_test_splits=3)
        with pytest.raises(ValueError):
            CombinatorialPurgedCV(n_splits=3, n_test_splits=5)

    def test_n_paths_count(self):
        """n_paths() = C(n_splits, n_test_splits)."""
        cv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2)
        assert cv.n_paths() == 15

        cv2 = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)
        assert cv2.n_paths() == 10


class TestCombinatorialPurgedCVSplit:
    def test_split_with_instant_labels(self):
        """Когда label_end_times=None — использует timestamps."""
        cv = CombinatorialPurgedCV(n_splits=4, n_test_splits=2, embargo_pct=0.01)
        ts = pd.date_range("2024-01-01", periods=400, freq="D")
        paths = list(cv.split(ts))
        # C(4, 2) = 6 путей
        assert len(paths) == 6

    def test_split_with_label_end_times(self):
        """С явно заданными label_end_times — purge работает."""
        cv = CombinatorialPurgedCV(n_splits=4, n_test_splits=2, embargo_pct=0.01)
        ts = pd.date_range("2024-01-01", periods=400, freq="D")
        # label_end = obs_start (instant labels)
        label_ends = pd.Series(ts, index=ts)
        paths = list(cv.split(ts, label_ends))
        assert len(paths) == 6

    def test_split_purges_overlapping_labels(self):
        """Train-точки с label, перекрывающим test window, удаляются из train."""
        cv = CombinatorialPurgedCV(n_splits=4, n_test_splits=1, embargo_pct=0.0)
        ts = pd.date_range("2024-01-01", periods=400, freq="D")
        # Каждая точка имеет label, который заканчивается далеко в будущем
        # → ВСЕ train-точки будут "перекрывать" test window → purged
        label_ends = pd.Series(ts + pd.Timedelta(days=100), index=ts)
        paths = list(cv.split(ts, label_ends))
        # Проверяем, что хотя бы один train_idx короче чем общее число точек
        for train_idx, test_idx in paths:
            assert len(train_idx) < 400
            assert len(train_idx) + len(test_idx) <= 400

    def test_embargo_truncates_at_end(self):
        """Embargo после последнего test-блока не выходит за пределы данных."""
        cv = CombinatorialPurgedCV(n_splits=3, n_test_splits=1, embargo_pct=0.5)
        ts = pd.date_range("2024-01-01", periods=300, freq="D")
        paths = list(cv.split(ts))
        # Если бы embargo вышел за пределы — был бы IndexError
        for train_idx, test_idx in paths:
            assert max(train_idx) < 300
            assert max(test_idx) < 300

    def test_split_test_indices_have_no_overlap(self):
        """В каждом пути test-блоки не пересекаются."""
        cv = CombinatorialPurgedCV(n_splits=4, n_test_splits=2)
        ts = pd.date_range("2024-01-01", periods=400, freq="D")
        paths = list(cv.split(ts))
        for train_idx, test_idx in paths:
            # test_idx не пересекается с train_idx
            assert len(set(train_idx) & set(test_idx)) == 0


class TestPurgedKfoldIndices:
    def test_basic_yields_n_splits(self):
        ts = pd.date_range("2024-01-01", periods=100, freq="D")
        label_ends = pd.Series(ts, index=ts)
        results = list(purged_kfold_indices(ts, label_ends, n_splits=5))
        assert len(results) == 5

    def test_train_test_no_overlap(self):
        ts = pd.date_range("2024-01-01", periods=100, freq="D")
        label_ends = pd.Series(ts, index=ts)
        for train_idx, test_idx in purged_kfold_indices(ts, label_ends, n_splits=5):
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_purges_overlapping_observations(self):
        """Train observations с label, перекрывающим test, удаляются."""
        ts = pd.date_range("2024-01-01", periods=100, freq="D")
        # Каждый label заканчивается на +50 дней — все перекрывают test
        label_ends = pd.Series(ts + pd.Timedelta(days=50), index=ts)
        for train_idx, test_idx in purged_kfold_indices(ts, label_ends, n_splits=5):
            # Должны быть purged — train_idx значительно короче
            assert len(train_idx) < 80  # baseline ~80 без purge

    def test_embargo_skips_post_test_observations(self):
        """После test-блока идёт embargo (skip) — train_idx не содержит post-test."""
        ts = pd.date_range("2024-01-01", periods=100, freq="D")
        label_ends = pd.Series(ts, index=ts)
        # embargo_pct=0.1 → skip 10 точек после test
        for train_idx, test_idx in purged_kfold_indices(
            ts, label_ends, n_splits=5, embargo_pct=0.1,
        ):
            test_max = max(test_idx)
            # Следующие 10 точек после test_max НЕ должны быть в train_idx
            for i in range(1, 11):
                if test_max + i < 100:
                    assert (test_max + i) not in train_idx

    def test_embargo_zero_no_gap(self):
        """embargo_pct=0 → нет gap после test."""
        ts = pd.date_range("2024-01-01", periods=100, freq="D")
        label_ends = pd.Series(ts, index=ts)
        for k, (train_idx, test_idx) in enumerate(
            purged_kfold_indices(ts, label_ends, n_splits=5, embargo_pct=0.0),
        ):
            _test_start = k * 20
            test_end = (k + 1) * 20 if k < 4 else 100
            # Train должен включать точки сразу после test (без gap)
            if test_end < 100:
                assert (test_end) in train_idx
