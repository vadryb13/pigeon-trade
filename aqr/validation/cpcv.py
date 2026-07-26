"""
Combinatorial Purged Cross-Validation (Lopez de Prado, 2018).

Advances in Financial Machine Learning, ch. 12.

Проблема стандартного k-fold CV на финансовых данных:
1. Labels пересекаются во времени (label at t зависит от future returns)
2. Train и test не iid → leakage

Решения:
- Purging: удалить из train наблюдения labels которых пересекаются с test window
- Embargo: добавить gap между train и test (для avoiding leakage через autocorrelation)
- Combinatorial: N*(N-1)/2 test splits вместо k, даёт больше out-of-sample paths
"""
from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations

import numpy as np
import pandas as pd


def _get_test_ranges(
    timestamps: pd.DatetimeIndex,
    n_splits: int,
    n_test_splits: int,
) -> list[list[tuple[int, int]]]:
    """
    Разбиваем timestamps на n_splits групп, выбираем C(n_splits, n_test_splits) сочетаний.

    Returns:
        List of test paths. Каждый path = список (start, end) tuples.
    """
    T = len(timestamps)
    if n_splits > T:
        raise ValueError(f"Not enough samples ({T}) for {n_splits} splits")

    fold_size = T // n_splits
    folds: list[tuple[int, int]] = []
    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_splits - 1 else T
        folds.append((start, end))

    paths = []
    for combo in combinations(range(n_splits), n_test_splits):
        paths.append([folds[i] for i in combo])
    return paths


def purged_kfold_indices(
    timestamps: pd.DatetimeIndex,
    label_end_times: pd.Series,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Purged K-Fold без combinatorial expansion — для быстрой валидации.

    Args:
        timestamps: индексы наблюдений (например, DatetimeIndex)
        label_end_times: Series индексированный timestamps, значения = время когда label становится известным
                         (для path-dependent labels типа triple-barrier)
        n_splits: количество фолдов
        embargo_pct: доля наблюдений которую отсечь после test window

    Yields:
        (train_idx, test_idx) — массивы позиций.
    """
    T = len(timestamps)
    embargo_size = int(T * embargo_pct)
    indices = np.arange(T)
    fold_size = T // n_splits

    for k in range(n_splits):
        test_start = k * fold_size
        test_end = (k + 1) * fold_size if k < n_splits - 1 else T
        test_idx = indices[test_start:test_end]
        test_time_start = timestamps[test_start]
        test_time_end = timestamps[test_end - 1]

        train_mask = np.ones(T, dtype=bool)
        train_mask[test_start:test_end] = False

        # Purge: убираем те train-точки чей label кончается ВНУТРИ test window
        for i in indices:
            if not train_mask[i]:
                continue
            label_end = label_end_times.iloc[i]
            obs_start = timestamps[i]
            if label_end >= test_time_start and obs_start <= test_time_end:
                train_mask[i] = False

        # Embargo после test-окна
        embargo_end = min(test_end + embargo_size, T)
        train_mask[test_end:embargo_end] = False

        train_idx = indices[train_mask]
        yield train_idx, test_idx


class CombinatorialPurgedCV:
    """
    CPCV: Lopez de Prado 2018, ch.12.

    Даёт множественные out-of-sample paths, что критично для strategy selection
    без look-ahead bias.

    Example:
        cv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2, embargo_pct=0.01)
        for train_idx, test_idx in cv.split(timestamps, label_end_times):
            fit on train, evaluate on test
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        embargo_pct: float = 0.01,
    ):
        if n_test_splits >= n_splits:
            raise ValueError("n_test_splits must be < n_splits")
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct

    def n_paths(self) -> int:
        from math import comb
        return comb(self.n_splits, self.n_test_splits)

    def split(
        self,
        timestamps: pd.DatetimeIndex,
        label_end_times: pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Yields (train_idx, test_idx) для каждой из C(n_splits, n_test_splits) комбинаций.
        """
        T = len(timestamps)
        indices = np.arange(T)
        embargo_size = int(T * self.embargo_pct)

        if label_end_times is None:
            # Если labels мгновенные — используем timestamps как end_times
            label_end_times = pd.Series(timestamps, index=timestamps)

        # Разбиваем на n_splits фолдов
        fold_size = T // self.n_splits
        folds = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else T
            folds.append((start, end))

        for test_combo in combinations(range(self.n_splits), self.n_test_splits):
            test_mask = np.zeros(T, dtype=bool)
            test_ranges = []
            for fi in test_combo:
                s, e = folds[fi]
                test_mask[s:e] = True
                test_ranges.append((s, e))

            train_mask = ~test_mask

            # Purge each train observation whose label overlaps any test window
            for s, e in test_ranges:
                t_start = timestamps[s]
                t_end = timestamps[e - 1]
                for i in indices:
                    if not train_mask[i]:
                        continue
                    obs_start = timestamps[i]
                    label_end = label_end_times.iloc[i] if i < len(label_end_times) else obs_start
                    if label_end >= t_start and obs_start <= t_end:
                        train_mask[i] = False

                # Embargo после каждого test-блока
                emb_end = min(e + embargo_size, T)
                train_mask[e:emb_end] = False

            train_idx = indices[train_mask]
            test_idx = indices[test_mask]
            yield train_idx, test_idx
