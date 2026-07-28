"""
Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

Стандартный Sharpe Ratio завышен когда:
1. Мы тестируем много стратегий (multiple testing)
2. Возвраты имеют skewness/kurtosis (не Gaussian)
3. Track record короткий

DSR корректирует все три и даёт вероятность что true Sharpe > 0.

References:
- Bailey, D. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio.
  Journal of Portfolio Management, 40(5), pp. 94-107.
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _sharpe(returns: np.ndarray, annualization: float = 252.0) -> float:
    """Annualized Sharpe Ratio (assumes returns in periodic units)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(annualization))


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    sr_benchmark: float = 0.0,
    annualization: float = 252.0,
) -> float:
    """
    Probabilistic Sharpe Ratio: P[true SR > sr_benchmark] given observed returns.

    Учитывает skewness и kurtosis (не Gaussian).

    Args:
        returns: массив периодических возвратов (не аннуализованных)
        sr_benchmark: пороговое значение аннуализованного SR
        annualization: коэффициент аннуализации (252 для daily, 12 для monthly)

    Returns:
        Вероятность в [0, 1].
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 3:
        return 0.5

    sr_obs = _sharpe(r, annualization)

    # Периодический benchmark для сравнения с периодическим SR
    sr_bench_periodic = sr_benchmark / np.sqrt(annualization)
    sr_obs_periodic = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else 0.0

    gamma3 = stats.skew(r, bias=False)
    gamma4 = stats.kurtosis(r, bias=False, fisher=True)  # excess kurtosis

    # Std от sample SR (Mertens 2002)
    denom_sq = 1.0 - gamma3 * sr_obs_periodic + (gamma4 / 4.0) * sr_obs_periodic ** 2
    if denom_sq <= 0:
        denom_sq = 1e-10
    sigma_sr = np.sqrt(denom_sq / (n - 1))

    z = (sr_obs_periodic - sr_bench_periodic) / sigma_sr
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    trial_sharpes: np.ndarray | None = None,
    annualization: float = 252.0,
) -> dict:
    """
    Deflated Sharpe Ratio: PSR при benchmark = expected max SR из N iid шумовых испытаний.

    Ключевая формула Bailey/Lopez de Prado:
        E[max SR] ≈ V^{1/2} * ((1 - γ)Φ^{-1}(1-1/N) + γ*Φ^{-1}(1-1/(N*e)))
    где V — дисперсия SR по trials, γ — Euler-Mascheroni ≈ 0.5772.

    Args:
        returns: возвраты кандидата
        n_trials: сколько стратегий было испытано (включая эту)
        trial_sharpes: массив аннуализованных SR всех испытаний (для точной V).
                       Если None — используется приближение V=1.
        annualization: 252 для daily

    Returns:
        {
            "sharpe": наблюдаемый annualized SR,
            "expected_max_sharpe": E[max SR] под H0 (шум),
            "deflated_sharpe": P[true SR > E[max SR под шумом]],
            "n_trials": N,
            "verdict": "significant" | "not_significant"
        }
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 3:
        return {
            "sharpe": 0.0,
            "expected_max_sharpe": 0.0,
            "deflated_sharpe": 0.5,
            "n_trials": n_trials,
            "verdict": "insufficient_data",
        }

    sr_obs = _sharpe(r, annualization)

    # V — variance of Sharpe estimates across trials
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        v = float(np.var(trial_sharpes, ddof=1))
    else:
        v = 1.0  # conservative

    euler_mascheroni = 0.5772156649
    n = max(int(n_trials), 1)
    _epsilon = 1e-12

    if n == 1:
        expected_max_sr = 0.0
    else:
        # Bailey & Lopez de Prado (2014), eq. (2)
        phi_inv_1 = stats.norm.ppf(1.0 - 1.0 / n)
        # Используем epsilon чтобы избежать ppf(1.0) → inf при больших n
        phi_inv_2 = stats.norm.ppf(1.0 - max(1.0 / (n * np.e), _epsilon))
        expected_max_sr = np.sqrt(v) * (
            (1.0 - euler_mascheroni) * phi_inv_1 + euler_mascheroni * phi_inv_2
        )

    dsr = probabilistic_sharpe_ratio(r, sr_benchmark=expected_max_sr, annualization=annualization)

    return {
        "sharpe": sr_obs,
        "expected_max_sharpe": float(expected_max_sr),
        "deflated_sharpe": float(dsr),
        "n_trials": n,
        "verdict": "significant" if dsr > 0.95 else "not_significant",
    }


def min_track_record_length(
    returns: np.ndarray,
    sr_target: float = 1.0,
    confidence: float = 0.95,
    annualization: float = 252.0,
) -> int:
    """
    Minimum Track Record Length (Bailey & Lopez de Prado, 2012).

    Сколько наблюдений нужно, чтобы отвергнуть H0: true SR <= sr_target
    с заданной confidence, при данном распределении возвратов.

    Returns:
        int — количество периодов (дней/месяцев в зависимости от annualization).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 3 or r.std(ddof=1) == 0:
        return -1

    sr_obs = r.mean() / r.std(ddof=1)
    sr_target_periodic = sr_target / np.sqrt(annualization)

    if sr_obs <= sr_target_periodic:
        return -1  # never reachable

    gamma3 = stats.skew(r, bias=False)
    gamma4 = stats.kurtosis(r, bias=False, fisher=True)

    z_alpha = stats.norm.ppf(confidence)
    numerator = 1.0 - gamma3 * sr_obs + (gamma4 / 4.0) * sr_obs ** 2
    denominator = (sr_obs - sr_target_periodic) ** 2

    if denominator <= 0:
        return -1

    min_len = 1.0 + numerator * (z_alpha ** 2) / denominator
    return int(np.ceil(min_len))
