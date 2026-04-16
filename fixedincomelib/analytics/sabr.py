from enum import Enum
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from scipy.stats import norm

from fixedincomelib.analytics.european_options import (
    CallOrPut,
    SimpleMetrics,
    EuropeanOptionAnalytics,
)


class SabrMetrics(Enum):
    ALPHA = "alpha"
    BETA = "beta"
    NU = "nu"
    RHO = "rho"
    DALPHA = "dalpha"
    DLNSIGMA = "dlnsigma"
    DNORMALSIGMA = "dnormalsigma"
    DBETA = "dbeta"
    DRHO = "drho"
    DNU = "dnu"
    DFORWARD = "dforward"
    DSTRIKE = "dstrike"
    DTTE = "dtte"
    DSTRIKESTRIKE = "dstrikestrike"
    D_LN_SIGMA_D_FORWARD = "d_ln_sigma_d_forward"
    D_LN_SIGMA_D_STRIKE = "d_ln_sigma_d_strike"
    D_LN_SIGMA_D_TTE = "d_ln_sigma_d_tte"
    D_LN_SIGMA_D_ALPHA = "d_ln_sigma_d_alpha"
    D_LN_SIGMA_D_BETA = "d_ln_sigma_d_beta"
    D_LN_SIGMA_D_NU = "d_ln_sigma_d_nu"
    D_LN_SIGMA_D_RHO = "d_ln_sigma_d_rho"
    D_LN_SIGMA_D_STRIKESTRIKE = "d_ln_sigma_d_strike_strike"
    D_ALPHA_D_LN_SIGMA_ATM = "d_alpha_d_ln_sigma_atm"
    D_ALPHA_D_FORWARD = "d_alpha_d_forward"
    D_ALPHA_D_TTE = "d_alpha_d_tte"
    D_ALPHA_D_BETA = "d_alpha_d_beta"
    D_ALPHA_D_NU = "d_alpha_d_nu"
    D_ALPHA_D_RHO = "d_alpha_d_rho"
    D_NORMAL_SIGMA_D_ALPHA = "d_normal_sigma_d_alpha"
    D_NORMAL_SIGMA_D_BETA = "d_normal_sigma_d_beta"
    D_NORMAL_SIGMA_D_NU = "d_normal_sigma_d_nu"
    D_NORMAL_SIGMA_D_RHO = "d_normal_sigma_d_rho"
    D_NORMAL_SIGMA_D_FORWARD = "d_normal_sigma_d_forward"
    D_NORMAL_SIGMA_D_TTE = "d_normal_sigma_d_tte"
    D_ALPHA_D_NORMAL_SIGMA_ATM = "d_alpha_d_normal_sigma_atm"

    @classmethod
    def from_string(cls, value: str) -> "SabrMetrics":
        return cls(value.lower())

    def to_string(self) -> str:
        return self.value


class SABRAnalytics:
    EPSILON = 1.0e-5

    @staticmethod
    def _validate(F, K, T, a, b, r, n):
        if F <= 0 or K <= 0:
            raise ValueError("shifted forward and strike must be positive")
        if T < 0:
            raise ValueError("time_to_expiry must be non-negative")
        if a <= 0:
            raise ValueError("alpha must be positive")
        if not (0.0 <= b <= 1.0):
            raise ValueError("beta must be in [0, 1]")
        if abs(r) >= 1.0:
            raise ValueError("rho must be in (-1, 1)")
        if n < 0:
            raise ValueError("nu must be non-negative")

    @staticmethod
    def _hagan_lognormal_vol(F: float, K: float, T: float, a: float, b: float, r: float, n: float) -> float:
        """Hagan et al. SABR lognormal implied volatility."""
        SABRAnalytics._validate(F, K, T, a, b, r, n)
        one_minus_b = 1.0 - b
        FK = F * K
        logFK = np.log(F / K)
        A = a / (FK ** (one_minus_b / 2.0))
        w1 = 1.0 + (one_minus_b**2 / 24.0) * logFK**2 + (one_minus_b**4 / 1920.0) * logFK**4
        w2 = (
            (one_minus_b**2 / 24.0) * a**2 / (FK ** one_minus_b)
            + 0.25 * r * b * n * a / (FK ** (one_minus_b / 2.0))
            + ((2.0 - 3.0 * r**2) / 24.0) * n**2
        )
        if abs(logFK) < 1.0e-12 or n == 0.0:
            z_over_x = 1.0
        else:
            z = (n / a) * (FK ** (one_minus_b / 2.0)) * logFK
            xz = np.log((np.sqrt(1.0 - 2.0 * r * z + z * z) + z - r) / (1.0 - r))
            z_over_x = z / xz if abs(xz) > 1.0e-14 else 1.0
        return float(A * z_over_x * w1 * (1.0 + w2 * T))

    @staticmethod
    def _central_diff(func, x, h=None):
        h = h or max(1.0e-6, abs(x) * 1.0e-5)
        if x - h <= 0:
            return (func(x + h) - func(x)) / h
        return (func(x + h) - func(x - h)) / (2.0 * h)

    @staticmethod
    def _vol_and_risk(F, K, T, a, b, r, n, calc_risk=False, z_cut=1e-2) -> Tuple[float, Dict[SabrMetrics, float]]:
        sigma = SABRAnalytics._hagan_lognormal_vol(F, K, T, a, b, r, n)
        risks: Dict[SabrMetrics, float] = {}
        if not calc_risk:
            return sigma, risks
        risks[SabrMetrics.D_LN_SIGMA_D_FORWARD] = SABRAnalytics._central_diff(lambda x: SABRAnalytics._hagan_lognormal_vol(x, K, T, a, b, r, n), F)
        risks[SabrMetrics.D_LN_SIGMA_D_STRIKE] = SABRAnalytics._central_diff(lambda x: SABRAnalytics._hagan_lognormal_vol(F, x, T, a, b, r, n), K)
        risks[SabrMetrics.D_LN_SIGMA_D_TTE] = (SABRAnalytics._hagan_lognormal_vol(F, K, T + 1e-5, a, b, r, n) - sigma) / 1e-5
        risks[SabrMetrics.D_LN_SIGMA_D_ALPHA] = SABRAnalytics._central_diff(lambda x: SABRAnalytics._hagan_lognormal_vol(F, K, T, x, b, r, n), a)
        risks[SabrMetrics.D_LN_SIGMA_D_BETA] = SABRAnalytics._central_diff(lambda x: SABRAnalytics._hagan_lognormal_vol(F, K, T, a, np.clip(x, 1e-8, 1 - 1e-8), r, n), b, h=1e-5)
        risks[SabrMetrics.D_LN_SIGMA_D_NU] = SABRAnalytics._central_diff(lambda x: SABRAnalytics._hagan_lognormal_vol(F, K, T, a, b, r, max(x, 0.0)), n, h=max(1e-6, n * 1e-5))
        risks[SabrMetrics.D_LN_SIGMA_D_RHO] = SABRAnalytics._central_diff(lambda x: SABRAnalytics._hagan_lognormal_vol(F, K, T, a, b, np.clip(x, -0.999999, 0.999999), n), r, h=1e-5)
        risks[SabrMetrics.D_LN_SIGMA_D_STRIKESTRIKE] = (
            SABRAnalytics._hagan_lognormal_vol(F, K + 1e-4, T, a, b, r, n) - 2.0 * sigma + SABRAnalytics._hagan_lognormal_vol(F, max(K - 1e-4, 1e-12), T, a, b, r, n)
        ) / (1e-4**2)
        return sigma, risks

    @staticmethod
    def lognormal_vol_from_alpha(forward, strike, time_to_expiry, alpha, beta, rho, nu, shift: Optional[float] = 0.0, calc_risk: Optional[bool] = False):
        res: Dict[Any, float] = {}
        vol, risks = SABRAnalytics._vol_and_risk(forward + shift, strike + shift, time_to_expiry, alpha, beta, rho, nu, calc_risk)
        res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL] = vol
        res.update(risks)
        return res

    @staticmethod
    def alpha_from_atm_lognormal_sigma(forward, time_to_expiry, sigma_atm_lognormal, beta, rho, nu, shift: Optional[float] = 0.0, calc_risk: Optional[bool] = False, max_iter: Optional[int] = 50, tol: Optional[float] = 1e-12):
        F = forward + shift
        target = sigma_atm_lognormal
        if F <= 0 or target <= 0:
            raise ValueError("shifted forward and target lognormal vol must be positive")
        alpha = max(target * F ** (1.0 - beta), 1e-8)
        lo, hi = 1e-12, max(1.0, 10.0 * alpha)
        while SABRAnalytics._hagan_lognormal_vol(F, F, time_to_expiry, hi, beta, rho, nu) < target:
            hi *= 2.0
        for _ in range(max_iter):
            f = SABRAnalytics._hagan_lognormal_vol(F, F, time_to_expiry, alpha, beta, rho, nu) - target
            if abs(f) < tol:
                break
            dfdalpha = SABRAnalytics._central_diff(lambda x: SABRAnalytics._hagan_lognormal_vol(F, F, time_to_expiry, x, beta, rho, nu), alpha)
            if dfdalpha <= 0 or not np.isfinite(dfdalpha):
                alpha_new = 0.5 * (lo + hi)
            else:
                alpha_new = alpha - f / dfdalpha
                if alpha_new <= lo or alpha_new >= hi or not np.isfinite(alpha_new):
                    alpha_new = 0.5 * (lo + hi)
            if f > 0:
                hi = alpha
            else:
                lo = alpha
            alpha = alpha_new
        res: Dict[SabrMetrics, float] = {SabrMetrics.ALPHA: float(alpha)}
        if calc_risk:
            dfa = SABRAnalytics._central_diff(lambda x: SABRAnalytics._hagan_lognormal_vol(F, F, time_to_expiry, x, beta, rho, nu), alpha)
            res[SabrMetrics.D_ALPHA_D_LN_SIGMA_ATM] = 1.0 / dfa
            for key, bump_func in [
                (SabrMetrics.D_ALPHA_D_FORWARD, lambda x: SABRAnalytics.alpha_from_atm_lognormal_sigma(x, time_to_expiry, target, beta, rho, nu, shift)[SabrMetrics.ALPHA]),
                (SabrMetrics.D_ALPHA_D_TTE, lambda x: SABRAnalytics.alpha_from_atm_lognormal_sigma(forward, x, target, beta, rho, nu, shift)[SabrMetrics.ALPHA]),
                (SabrMetrics.D_ALPHA_D_BETA, lambda x: SABRAnalytics.alpha_from_atm_lognormal_sigma(forward, time_to_expiry, target, x, rho, nu, shift)[SabrMetrics.ALPHA]),
                (SabrMetrics.D_ALPHA_D_NU, lambda x: SABRAnalytics.alpha_from_atm_lognormal_sigma(forward, time_to_expiry, target, beta, rho, max(x, 0), shift)[SabrMetrics.ALPHA]),
                (SabrMetrics.D_ALPHA_D_RHO, lambda x: SABRAnalytics.alpha_from_atm_lognormal_sigma(forward, time_to_expiry, target, beta, np.clip(x, -0.999, 0.999), nu, shift)[SabrMetrics.ALPHA]),
            ]:
                base_x = {SabrMetrics.D_ALPHA_D_FORWARD: forward, SabrMetrics.D_ALPHA_D_TTE: time_to_expiry, SabrMetrics.D_ALPHA_D_BETA: beta, SabrMetrics.D_ALPHA_D_NU: nu, SabrMetrics.D_ALPHA_D_RHO: rho}[key]
                res[key] = SABRAnalytics._central_diff(bump_func, base_x, h=max(1e-5, abs(base_x) * 1e-5))
        return res

    @staticmethod
    def atm_normal_sigma_from_alpha(forward, time_to_expiry, alpha, beta, rho, nu, shift: Optional[float] = 0.0, calc_risk: Optional[bool] = False, tol: Optional[float] = 1e-8):
        ln_vol = SABRAnalytics.lognormal_vol_from_alpha(forward, forward, time_to_expiry, alpha, beta, rho, nu, shift, False)[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
        res = EuropeanOptionAnalytics.lognormal_vol_to_normal_vol(forward, forward, time_to_expiry, ln_vol, calc_risk, shift, tol)
        out = {SimpleMetrics.IMPLIED_NORMAL_VOL: res[SimpleMetrics.IMPLIED_NORMAL_VOL]}
        if calc_risk:
            out[SabrMetrics.D_NORMAL_SIGMA_D_ALPHA] = SABRAnalytics._central_diff(lambda x: SABRAnalytics.atm_normal_sigma_from_alpha(forward, time_to_expiry, x, beta, rho, nu, shift)[SimpleMetrics.IMPLIED_NORMAL_VOL], alpha)
            out[SabrMetrics.D_NORMAL_SIGMA_D_BETA] = SABRAnalytics._central_diff(lambda x: SABRAnalytics.atm_normal_sigma_from_alpha(forward, time_to_expiry, alpha, x, rho, nu, shift)[SimpleMetrics.IMPLIED_NORMAL_VOL], beta, h=1e-5)
            out[SabrMetrics.D_NORMAL_SIGMA_D_NU] = SABRAnalytics._central_diff(lambda x: SABRAnalytics.atm_normal_sigma_from_alpha(forward, time_to_expiry, alpha, beta, rho, max(x, 0), shift)[SimpleMetrics.IMPLIED_NORMAL_VOL], nu)
            out[SabrMetrics.D_NORMAL_SIGMA_D_RHO] = SABRAnalytics._central_diff(lambda x: SABRAnalytics.atm_normal_sigma_from_alpha(forward, time_to_expiry, alpha, beta, np.clip(x, -0.999, 0.999), nu, shift)[SimpleMetrics.IMPLIED_NORMAL_VOL], rho, h=1e-5)
            out[SabrMetrics.D_NORMAL_SIGMA_D_FORWARD] = SABRAnalytics._central_diff(lambda x: SABRAnalytics.atm_normal_sigma_from_alpha(x, time_to_expiry, alpha, beta, rho, nu, shift)[SimpleMetrics.IMPLIED_NORMAL_VOL], forward)
            out[SabrMetrics.D_NORMAL_SIGMA_D_TTE] = SABRAnalytics._central_diff(lambda x: SABRAnalytics.atm_normal_sigma_from_alpha(forward, x, alpha, beta, rho, nu, shift)[SimpleMetrics.IMPLIED_NORMAL_VOL], time_to_expiry)
        return out

    @staticmethod
    def alpha_from_atm_normal_sigma(forward, time_to_expiry, sigma_atm_normal, beta, rho, nu, shift: Optional[float] = 0.0, calc_risk: bool = False, max_iter: int = 50, tol: float = 1e-8):
        ln_res = EuropeanOptionAnalytics.normal_vol_to_lognormal_vol(forward, forward, time_to_expiry, sigma_atm_normal, calc_risk, shift, tol)
        ln_vol = ln_res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
        res = SABRAnalytics.alpha_from_atm_lognormal_sigma(forward, time_to_expiry, ln_vol, beta, rho, nu, shift, calc_risk, max_iter, tol)
        if calc_risk:
            d_alpha_d_ln = res[SabrMetrics.D_ALPHA_D_LN_SIGMA_ATM]
            res[SabrMetrics.D_ALPHA_D_NORMAL_SIGMA_ATM] = d_alpha_d_ln * ln_res[SimpleMetrics.D_LN_VOL_D_N_VOL]
            res[SabrMetrics.D_ALPHA_D_FORWARD] += d_alpha_d_ln * ln_res[SimpleMetrics.D_LN_VOL_D_FORWARD]
            res[SabrMetrics.D_ALPHA_D_TTE] += d_alpha_d_ln * ln_res[SimpleMetrics.D_LN_VOL_D_TTE]
        return res

    @staticmethod
    def european_option_alpha(forward, strike, time_to_expiry, opt_type, alpha, beta, rho, nu, shift: Optional[float] = 0.0, calc_risk: Optional[bool] = False):
        ln = SABRAnalytics.lognormal_vol_from_alpha(forward, strike, time_to_expiry, alpha, beta, rho, nu, shift, calc_risk)
        iv = ln[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
        val = EuropeanOptionAnalytics.european_option_log_normal(forward + shift, strike + shift, time_to_expiry, iv, opt_type, calc_risk)
        if calc_risk:
            vega = val.pop(SimpleMetrics.VEGA)
            val[SimpleMetrics.DELTA] += vega * ln[SabrMetrics.D_LN_SIGMA_D_FORWARD]
            val[SimpleMetrics.THETA] -= vega * ln[SabrMetrics.D_LN_SIGMA_D_TTE]
            val[SimpleMetrics.STRIKE_RISK] += vega * ln[SabrMetrics.D_LN_SIGMA_D_STRIKE]
            val[SabrMetrics.DALPHA] = vega * ln[SabrMetrics.D_LN_SIGMA_D_ALPHA]
            val[SabrMetrics.DBETA] = vega * ln[SabrMetrics.D_LN_SIGMA_D_BETA]
            val[SabrMetrics.DRHO] = vega * ln[SabrMetrics.D_LN_SIGMA_D_RHO]
            val[SabrMetrics.DNU] = vega * ln[SabrMetrics.D_LN_SIGMA_D_NU]
            base = val[SimpleMetrics.PV]
            h = SABRAnalytics.EPSILON
            up = SABRAnalytics.european_option_alpha(forward + h, strike, time_to_expiry, opt_type, alpha, beta, rho, nu, shift)[SimpleMetrics.PV]
            dn = SABRAnalytics.european_option_alpha(forward - h, strike, time_to_expiry, opt_type, alpha, beta, rho, nu, shift)[SimpleMetrics.PV]
            val[SimpleMetrics.GAMMA] = (up - 2 * base + dn) / h**2
            kup = SABRAnalytics.european_option_alpha(forward, strike + h, time_to_expiry, opt_type, alpha, beta, rho, nu, shift)[SimpleMetrics.PV]
            kdn = SABRAnalytics.european_option_alpha(forward, strike - h, time_to_expiry, opt_type, alpha, beta, rho, nu, shift)[SimpleMetrics.PV]
            val[SimpleMetrics.STRIKE_RISK_2] = (kup - 2 * base + kdn) / h**2
        return val

    @staticmethod
    def european_option_ln_sigma(forward, strike, time_to_expiry, opt_type, ln_sigma_atm, beta, rho, nu, shift: Optional[float] = 0.0, calc_risk: Optional[bool] = False):
        alpha_res = SABRAnalytics.alpha_from_atm_lognormal_sigma(forward, time_to_expiry, ln_sigma_atm, beta, rho, nu, shift, calc_risk)
        alpha = alpha_res[SabrMetrics.ALPHA]
        val = SABRAnalytics.european_option_alpha(forward, strike, time_to_expiry, opt_type, alpha, beta, rho, nu, shift, calc_risk)
        if calc_risk:
            dvdalpha = val.pop(SabrMetrics.DALPHA)
            val[SabrMetrics.DLNSIGMA] = dvdalpha * alpha_res[SabrMetrics.D_ALPHA_D_LN_SIGMA_ATM]
            val[SimpleMetrics.DELTA] += dvdalpha * alpha_res[SabrMetrics.D_ALPHA_D_FORWARD]
            val[SimpleMetrics.THETA] -= dvdalpha * alpha_res[SabrMetrics.D_ALPHA_D_TTE]
            val[SabrMetrics.DBETA] += dvdalpha * alpha_res[SabrMetrics.D_ALPHA_D_BETA]
            val[SabrMetrics.DRHO] += dvdalpha * alpha_res[SabrMetrics.D_ALPHA_D_RHO]
            val[SabrMetrics.DNU] += dvdalpha * alpha_res[SabrMetrics.D_ALPHA_D_NU]
        return val

    @staticmethod
    def european_option_normal_sigma(forward, strike, time_to_expiry, opt_type, normal_sigma_atm, beta, rho, nu, shift: Optional[float] = 0.0, calc_risk: Optional[bool] = False):
        alpha_res = SABRAnalytics.alpha_from_atm_normal_sigma(forward, time_to_expiry, normal_sigma_atm, beta, rho, nu, shift, calc_risk)
        alpha = alpha_res[SabrMetrics.ALPHA]
        val = SABRAnalytics.european_option_alpha(forward, strike, time_to_expiry, opt_type, alpha, beta, rho, nu, shift, calc_risk)
        if calc_risk:
            dvdalpha = val.pop(SabrMetrics.DALPHA)
            val[SabrMetrics.DNORMALSIGMA] = dvdalpha * alpha_res[SabrMetrics.D_ALPHA_D_NORMAL_SIGMA_ATM]
            val[SimpleMetrics.DELTA] += dvdalpha * alpha_res[SabrMetrics.D_ALPHA_D_FORWARD]
            val[SimpleMetrics.THETA] -= dvdalpha * alpha_res[SabrMetrics.D_ALPHA_D_TTE]
            val[SabrMetrics.DBETA] += dvdalpha * alpha_res[SabrMetrics.D_ALPHA_D_BETA]
            val[SabrMetrics.DRHO] += dvdalpha * alpha_res[SabrMetrics.D_ALPHA_D_RHO]
            val[SabrMetrics.DNU] += dvdalpha * alpha_res[SabrMetrics.D_ALPHA_D_NU]
        return val

    @staticmethod
    def pdf_and_cdf(forward, time_to_expiry, alpha, beta, rho, nu, grids: List | np.ndarray, shift: Optional[float] = 0):
        xs = np.asarray(grids, dtype=float)
        dx = np.gradient(xs)
        call_prices = np.array([SABRAnalytics.european_option_alpha(forward, x, time_to_expiry, CallOrPut.CALL, alpha, beta, rho, nu, shift)[SimpleMetrics.PV] for x in xs])
        dC_dK = np.gradient(call_prices, xs)
        d2C_dK2 = np.gradient(dC_dK, xs)
        pdf = np.maximum(d2C_dK2, 0.0)
        if np.trapz(pdf, xs) > 0:
            pdf = pdf / np.trapz(pdf, xs)
        cdf = np.cumsum(pdf * dx)
        cdf = np.clip(cdf / max(cdf[-1], 1e-16), 0.0, 1.0)
        return xs, xs + shift, cdf, pdf

    @staticmethod
    def simulate_terminal_distribution(forward, time_to_expiry, alpha, beta, rho, nu, n_steps=100, n_paths=10000, shift: Optional[float] = 0.0, seed: Optional[int] = 1234):
        rng = np.random.default_rng(seed)
        dt = time_to_expiry / n_steps
        sqrt_dt = np.sqrt(dt)
        F = np.full(n_paths, forward + shift, dtype=float)
        sig = np.full(n_paths, alpha, dtype=float)
        for _ in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            dW1 = sqrt_dt * z1
            dW2 = sqrt_dt * (rho * z1 + np.sqrt(max(1 - rho * rho, 0.0)) * z2)
            F = np.maximum(F + sig * np.maximum(F, 1e-12) ** beta * dW1, 1e-12)
            sig = sig * np.exp(nu * dW2 - 0.5 * nu * nu * dt)
        return F - shift
