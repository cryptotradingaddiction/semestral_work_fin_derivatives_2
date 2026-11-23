#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1BP451 Financial Derivatives II - Project
Author: Teodor Tsohla
Valuation of the Bonus Outperformance Certificate on iShares 20+ Year Treasury Bond ETF (TLT),
ISIN CH1506044580, issued by Banque Internationale a Luxembourg S.A. (BIL).

Methodology (strictly aligned with course lectures Part 1 and Part 2):
  1. Payoff decomposition into a zero-coupon bond, binary call, and 22.163 European calls.
  2. Risk-neutral valuation via Black-Scholes / Black's formula with continuous dividend yield q.
  3. GARCH(1,1) volatility calibrated by Maximum Likelihood + expected average variance over option life.
  4. Recombining binomial tree (2 steps/day) for the issuer's 23-date Bermudan call option.

Data sources:
  * TLT prices, adjusted returns, and trailing 12m dividends: Yahoo Finance (yfinance)
  * 1-Year (DGS1) and 2-Year (DGS2) U.S. Treasury yields: FRED
  * Secondary-market quotes: Leonteq website
"""

from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm

# =====================================================================
# 1) CONTRACT PARAMETERS FROM TERMSHEET (ISIN CH1506044580)
# =====================================================================
DENOM = 1000.0                           # Denomination (USD 1,000)
INIT_FIX = 90.24                         # Initial Fixing Level & maturity-only barrier H (USD)
BONUS_LVL = 1.086                        # Bonus Level (108.60%)
STRIKE_BONUS = INIT_FIX * BONUS_LVL      # Outperformance call strike: 98.00064 USD
PARTICIPATION = 2.0                      # Upside participation above Bonus Level (200%)
EARLY_REDEMP_AMT = 1086.0                # Cash settlement upon issuer early call (USD)
BONUS_COUPON = EARLY_REDEMP_AMT - DENOM  # Binary bonus coupon (USD 86)
CALL_UNITS = PARTICIPATION * DENOM / INIT_FIX  # 2000 / 90.24 = 22.16312057 European calls
FINAL_FIX_DATE = date(2027, 11, 26)      # Final Fixing Date
REDEMP_DATE = date(2027, 12, 3)          # Final Redemption Date

# All 23 (Early Redemption Observation Date, Early Redemption Date) pairs from termsheet
OBS_SCHEDULE = [
    (date(2025, 12, 26), date(2026, 1, 6)),
    (date(2026, 1, 26), date(2026, 1, 30)),
    (date(2026, 2, 25), date(2026, 3, 4)),
    (date(2026, 3, 25), date(2026, 4, 1)),
    (date(2026, 4, 27), date(2026, 5, 4)),
    (date(2026, 5, 26), date(2026, 6, 1)),
    (date(2026, 6, 25), date(2026, 7, 2)),
    (date(2026, 7, 27), date(2026, 7, 31)),
    (date(2026, 8, 25), date(2026, 9, 1)),
    (date(2026, 9, 25), date(2026, 10, 2)),
    (date(2026, 10, 26), date(2026, 10, 30)),
    (date(2026, 11, 25), date(2026, 12, 2)),
    (date(2026, 12, 28), date(2027, 1, 4)),
    (date(2027, 1, 25), date(2027, 2, 1)),
    (date(2027, 2, 25), date(2027, 3, 4)),
    (date(2027, 3, 25), date(2027, 4, 5)),
    (date(2027, 4, 26), date(2027, 4, 30)),
    (date(2027, 5, 25), date(2027, 6, 1)),
    (date(2027, 6, 25), date(2027, 7, 2)),
    (date(2027, 7, 26), date(2027, 7, 30)),
    (date(2027, 8, 25), date(2027, 9, 1)),
    (date(2027, 9, 27), date(2027, 10, 1)),
    (date(2027, 10, 25), date(2027, 11, 1)),
]

# =====================================================================
# 2) VALUATION DATES & OBSERVED LEONTEQ QUOTES
# =====================================================================
VALUATION_DATES = [
    ("A", date(2025, 12, 3)),   # Issue Date
    ("B", date(2026, 10, 23)),  # Required Valuation Date
    ("C", date(2026, 11, 20)),  # Current Valuation Date
]

LEONTEQ_QUOTES = {  # Indicative quotes observed on Leonteq (% of notional)
    "A": 98.62,     # 3 Dec 2025
    "B": 93.97,     # 23 Oct 2026 (12:00 CET)
    "C": 93.79,     # 20 Nov 2026 (12:00 CET)
}

EST_WINDOW = 1260   # Estimation window for GARCH(1,1) MLE (approx. 5 years of daily returns)
OUTPUT_TEX = "outputs.tex"


# =====================================================================
# 3) VALUATION MODELS (BLACK-SCHOLES / BLACK + BINOMIAL TREE)
# =====================================================================
def year_frac(d1, d2):
    """ACT/365 year fraction between calendar dates d1 and d2."""
    return (d2 - d1).days / 365.0


def non_callable_value(S, r, q, sig, t0):
    """Closed-form risk-free valuation of the non-callable payoff (Eq. 4)."""
    tf = year_frac(t0, FINAL_FIX_DATE)
    tp = year_frac(t0, REDEMP_DATE)
    df = np.exp(-r * tp)
    fwd = S * np.exp((r - q) * tf)
    vol_t = sig * np.sqrt(tf)

    d2_bar = (np.log(fwd / INIT_FIX) - 0.5 * vol_t ** 2) / vol_t
    d1_call = (np.log(fwd / STRIKE_BONUS) + 0.5 * vol_t ** 2) / vol_t
    d2_call = d1_call - vol_t

    bond = DENOM * df
    digital = BONUS_COUPON * df * norm.cdf(d2_bar)
    call = CALL_UNITS * df * (fwd * norm.cdf(d1_call) - STRIKE_BONUS * norm.cdf(d2_call))

    return dict(
        fwd=float(fwd),
        prob_bonus=float(norm.cdf(d2_bar)),
        bond=float(bond),
        digital=float(digital),
        call=float(call),
        nc=float(bond + digital + call),
    )


def binomial_tree_valuation(S, r, q, sig, t0, steps_per_day=2, callable_flag=True):
    """Recombining binomial tree valuation for the 23-date Bermudan callable certificate."""
    n_steps = (FINAL_FIX_DATE - t0).days * steps_per_day
    dt = 1.0 / (365.0 * steps_per_day)
    u = np.exp(sig * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Terminal TLT prices and discounted maturity payoff at Final Fixing Date
    i = np.arange(n_steps + 1)
    S_T = S * (u ** (n_steps - 2 * i))
    payoff_T = DENOM + BONUS_COUPON * (S_T > INIT_FIX) + CALL_UNITS * np.maximum(S_T - STRIKE_BONUS, 0.0)
    V = np.exp(-r * year_frac(FINAL_FIX_DATE, REDEMP_DATE)) * payoff_T

    # Map future observation dates to (observation index, present value of USD 1,086 call payout)
    obs_map = {
        (o - t0).days * steps_per_day: (k + 1, EARLY_REDEMP_AMT * np.exp(-r * year_frac(o, pay)))
        for k, (o, pay) in enumerate(OBS_SCHEDULE) if o > t0
    }

    # Backward induction from maturity to valuation date t0
    called_dates = []
    for j in range(n_steps - 1, -1, -1):
        V = disc * (p * V[:j + 1] + (1.0 - p) * V[1:j + 2])
        if callable_flag and j in obs_map:
            k_num, call_pv = obs_map[j]
            if np.any(call_pv < V):
                called_dates.append(k_num)
            V = np.minimum(V, call_pv)

    called_dates.sort()
    return float(V[0]), called_dates


# =====================================================================
# 4) VOLATILITY ESTIMATION (HISTORICAL & GARCH(1,1) MLE)
# =====================================================================
def vol_historical(u, window=252):
    """1-year (252-day) equally weighted annualized historical volatility."""
    return float(np.std(u[-window:], ddof=1) * np.sqrt(252.0))


def fit_garch_mle(u):
    """Estimate GARCH(1,1) parameters by maximizing Gaussian log-likelihood (Part 1, slide 51)."""
    x = np.asarray(u, float) * 100.0  # Scale returns to % for numerical conditioning
    v0 = float(np.var(x))

    def neg_log_lik(params):
        omega, alpha, beta = params
        if omega <= 1e-8 or alpha <= 0.0 or beta <= 0.0 or (alpha + beta) >= 0.9995:
            return 1e12
        s2 = v0
        total = 0.0
        for xi in x:
            total += np.log(s2) + (xi * xi) / s2
            s2 = omega + alpha * (xi * xi) + beta * s2
        return total

    best = None
    for a0, b0 in [(0.05, 0.90), (0.03, 0.95), (0.08, 0.88)]:
        w0 = v0 * (1.0 - a0 - b0)
        res = minimize(
            neg_log_lik, [w0, a0, b0], method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-7},
        )
        if best is None or res.fun < best.fun:
            best = res

    w_pct, a_hat, b_hat = best.x
    w_d = w_pct / 1e4
    vl_d = w_d / (1.0 - a_hat - b_hat)

    # Filter conditional daily variance forward to the valuation date
    s2 = v0 / 1e4
    for xi in u:
        s2 = w_d + a_hat * (xi * xi) + b_hat * s2
    return dict(omega=w_d, alpha=a_hat, beta=b_hat, vl=vl_d, s2n=s2, n=len(u))


def garch_option_vol(alpha, beta, vl, s2n, n_days):
    """Expected annualized volatility over remaining option life of n_days (Part 1, slides 52-53)."""
    k = np.arange(n_days)
    expected_vars = vl + ((alpha + beta) ** k) * (s2n - vl)
    return float(np.sqrt(252.0 * np.mean(expected_vars)))


# =====================================================================
# 5) LIVE MARKET DATA DOWNLOAD (YAHOO FINANCE + FRED)
# =====================================================================
def fetch_tlt_data():
    """Download TLT daily prices, dividend-adjusted log-returns, and cash dividends from Yahoo Finance."""
    df = yf.download("TLT", start="2015-01-01", auto_adjust=False, progress=False)
    if hasattr(df.columns, "droplevel") and df.columns.nlevels > 1:
        df.columns = df.columns.droplevel(1)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()

    adj = df["Adj Close"].squeeze().dropna()
    close = df["Close"].squeeze().dropna()
    log_rets = np.log(adj / adj.shift(1)).dropna()

    divs = yf.Ticker("TLT").dividends
    if getattr(divs.index, "tz", None) is not None:
        divs.index = divs.index.tz_localize(None)
    divs = divs.sort_index()
    return log_rets, close, divs


def fetch_fred_yields():
    """Download daily 1-Year (DGS1) and 2-Year (DGS2) U.S. Treasury yields from FRED."""
    d1 = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1", index_col=0, parse_dates=True)
    d2 = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2", index_col=0, parse_dates=True)
    s1 = pd.to_numeric(d1.iloc[:, 0], errors="coerce").dropna().sort_index() / 100.0
    s2 = pd.to_numeric(d2.iloc[:, 0], errors="coerce").dropna().sort_index() / 100.0
    return s1, s2


def get_continuous_rf_rate(t0, dgs1, dgs2):
    """Linearly interpolate 1Y/2Y Treasury yields to tau_pay and convert to continuous compounding."""
    ts = pd.Timestamp(t0)
    y1 = float(dgs1[dgs1.index <= ts].iloc[-1])
    y2 = float(dgs2[dgs2.index <= ts].iloc[-1])
    tp = year_frac(t0, REDEMP_DATE)
    w = float(np.clip(tp - 1.0, 0.0, 1.0))
    y_par = (1.0 - w) * y1 + w * y2
    r_cont = float(2.0 * np.log(1.0 + y_par / 2.0))
    return r_cont, y1, y2


# =====================================================================
# 6) LATEX EXPORT HELPERS
# =====================================================================
def fmt_num(x, nd=2):
    if x < 0:
        return "$-$" + fmt_num(-x, nd)
    return f"{x:,.{nd}f}".replace(",", r"\,")


def fmt_pct(x, nd=2):
    return fmt_num(100.0 * x, nd) + r"\,\%"


def fmt_date(d):
    return d.strftime("%d %b %Y").lstrip("0")


def fmt_sci(x):
    m, e = f"{x:.3e}".split("e")
    return f"{m}\\cdot 10^{{{int(e)}}}"


def write_latex(out_path, results):
    """Write all scalar macros and LaTeX tables consumed by main.tex."""
    lines = []
    add = lines.append
    keys = list(results)

    def macro(name, body):
        add(f"\\newcommand{{\\{name}}}{{{body}}}")

    for k in keys:
        i = results[k]
        nc = i["nc_res"]
        m_pct = i["tree_call"] / 10.0
        qt = i["quote"]
        macro(f"date{k}", fmt_date(i["val_date"]))
        macro(f"S{k}", fmt_num(i["S"]))
        macro(f"r{k}", fmt_pct(i["r"]))
        macro(f"q{k}", fmt_pct(i["q"]))
        macro(f"sigg{k}", fmt_pct(i["sig_garch"]))
        macro(f"sign{k}", fmt_pct(i["sig_n"]))
        macro(f"sigvl{k}", fmt_pct(i["sig_vl"]))
        macro(f"nc{k}", fmt_num(nc["nc"]))
        macro(f"ncp{k}", fmt_pct(nc["nc"] / DENOM))
        macro(f"call{k}", fmt_num(i["issuer_call"]))
        macro(f"c{k}", fmt_pct(i["tree_call"] / DENOM))
        macro(f"cusd{k}", fmt_num(i["tree_call"]))
        macro(f"qt{k}", fmt_pct(qt / 100.0))
        macro(f"gap{k}", fmt_num(100.0 - m_pct))
        macro(f"diff{k}", fmt_num(m_pct - qt))
        macro(f"exobs{k}", ", ".join(str(x) for x in i["called_dates"]) or "none")

    col_spec = "c" * len(keys)
    header = " & ".join(fmt_date(results[k]["val_date"]) for k in keys)

    def row(label, fn):
        return f"{label} & " + " & ".join(fn(results[k]) for k in keys) + r" \\"

    def fmt_call_dates(cd):
        if not cd:
            return "none"
        return ", ".join(f"{fmt_date(OBS_SCHEDULE[idx - 1][0])} (no.\\ {idx})" for idx in cd)

    # Table 2: Market inputs and volatility estimates
    add(r"\newcommand{\TabInputs}{\begin{table}[ht]\centering\small")
    add(r"\caption{Market inputs (Yahoo Finance, FRED) and volatility estimates.}\label{tab:inputs}")
    add(f"\\begin{{tabular}}{{@{{}}l{col_spec}@{{}}}}\\toprule")
    add(f" & {header} \\\\\\midrule")
    add(row("TLT close price $S_t$ (USD)", lambda i: fmt_num(i["S"])))
    add(row(r"Time to final fixing $\tau_{\rm fix}$ (yr) / $N$ days",
            lambda i: f"{fmt_num(year_frac(i['val_date'], FINAL_FIX_DATE), 3)} / {i['N']}"))
    add(row(r"Time to payment $\tau_{\rm pay}$ (yr)", lambda i: fmt_num(year_frac(i["val_date"], REDEMP_DATE), 3)))
    add(row("FRED Treasury 1Y / 2Y yield", lambda i: f"{fmt_pct(i['y1'])} / {fmt_pct(i['y2'])}"))
    add(row("Risk-free rate $r$ (continuous)", lambda i: fmt_pct(i["r"])))
    add(row("Dividend yield $q$ (trailing 12m)", lambda i: fmt_pct(i["q"])))
    add(r"\midrule")
    add(row("GARCH(1,1) expected volatility (over $N$ days)", lambda i: fmt_pct(i["sig_garch"])))
    add(row("Historical volatility (last 252 days)", lambda i: fmt_pct(i["sig_hist"])))
    add(r"\bottomrule\end{tabular}\end{table}}")

    # Table 3: GARCH(1,1) MLE parameters
    add(r"\newcommand{\TabGarch}{\begin{table}[ht]\centering\small")
    add(r"\caption{GARCH(1,1) parameters estimated by Maximum Likelihood (1,260 daily log-returns).}\label{tab:garch}")
    add(f"\\begin{{tabular}}{{@{{}}l{col_spec}@{{}}}}\\toprule")
    add(f" & {header} \\\\\\midrule")
    add(row(r"$\omega$", lambda i: "$" + fmt_sci(i["g"]["omega"]) + "$"))
    add(row(r"$\alpha$", lambda i: fmt_num(i["g"]["alpha"], 4)))
    add(row(r"$\beta$", lambda i: fmt_num(i["g"]["beta"], 4)))
    add(row(r"$\gamma = 1 - \alpha - \beta$", lambda i: fmt_num(1.0 - i["g"]["alpha"] - i["g"]["beta"], 4)))
    add(row(r"Long-run volatility $\sqrt{252\,V_L}$", lambda i: fmt_pct(i["sig_vl"])))
    add(row(r"Next-day volatility $\sigma_n$", lambda i: fmt_pct(i["sig_n"])))
    add(row(r"Expected volatility over option life", lambda i: fmt_pct(i["sig_garch"])))
    add(r"\bottomrule\end{tabular}\end{table}}")

    # Table 4: Non-callable Black-Scholes / Black valuation
    add(r"\newcommand{\TabNonCallable}{\begin{table}[ht]\centering\small")
    add(r"\caption{Valuation without the issuer's call option (USD per 1,000 USD certificate).}\label{tab:nc}")
    add(f"\\begin{{tabular}}{{@{{}}l{col_spec}@{{}}}}\\toprule")
    add(f" & {header} \\\\\\midrule")
    add(row("Forward TLT price $F_0$ (USD)", lambda i: fmt_num(i["nc_res"]["fwd"])))
    add(row(r"Risk-neutral prob.\ $\mathbb{Q}(S_T > 90.24)$", lambda i: fmt_pct(i["nc_res"]["prob_bonus"], 1)))
    add(r"\midrule")
    add(row("Zero-coupon bond (USD 1,000)", lambda i: fmt_num(i["nc_res"]["bond"])))
    add(row("Binary call (USD 86, $K = 90.24$)", lambda i: fmt_num(i["nc_res"]["digital"])))
    add(row("22.163 European calls ($K_b = 98.00$)", lambda i: fmt_num(i["nc_res"]["call"])))
    add(r"\midrule")
    add(row(r"Total non-callable value $V^{\rm NC}$ (USD)", lambda i: fmt_num(i["nc_res"]["nc"])))
    add(row(r"Total non-callable value (\% of notional)", lambda i: fmt_pct(i["nc_res"]["nc"] / DENOM)))
    add(r"\bottomrule\end{tabular}\end{table}}")

    # Table 5: Binomial tree valuation and comparison with Leonteq quotes
    add(r"\newcommand{\TabTree}{\begin{table}[ht]\centering\small")
    add(r"\caption{Binomial tree valuation (2 steps/day) and comparison with Leonteq quotes.}\label{tab:tree}")
    add(f"\\begin{{tabular}}{{@{{}}l{col_spec}@{{}}}}\\toprule")
    add(f" & {header} \\\\\\midrule")
    add(row(r"Closed-form non-callable $V^{\rm NC}$ (USD)", lambda i: fmt_num(i["nc_res"]["nc"])))
    add(row("Binomial tree without issuer call (USD)", lambda i: fmt_num(i["tree_nc"])))
    add(row("Binomial tree with issuer call (USD)", lambda i: fmt_num(i["tree_call"])))
    add(row("Value of issuer's Bermudan call (USD)", lambda i: fmt_num(i["issuer_call"])))
    add(row("Call date(s) with optimal exercise", lambda i: fmt_call_dates(i["called_dates"])))
    add(r"\midrule")
    add(row(r"\textbf{Model certificate value (\% notional)}", lambda i: r"\textbf{" + fmt_pct(i["tree_call"] / DENOM) + "}"))
    add(row(r"Leonteq quote (\% notional)", lambda i: fmt_pct(i["quote"] / 100.0)))
    add(row("Difference: Model $-$ Leonteq (pp)", lambda i: fmt_num(i["tree_call"] / 10.0 - i["quote"])))
    add(r"\bottomrule\end{tabular}\end{table}}")

    # Table 6: Sensitivity analysis
    add(r"\newcommand{\TabSens}{\begin{table}[ht]\centering\small")
    add(r"\caption{Sensitivity of the callable certificate value (\% of notional).}\label{tab:sens}")
    add(f"\\begin{{tabular}}{{@{{}}l{col_spec}@{{}}}}\\toprule")
    add(f" & {header} \\\\\\midrule")
    for j in range(len(results[keys[0]]["sens"])):
        lbl = results[keys[0]]["sens"][j][0]
        add(f"{lbl} & " + " & ".join(fmt_pct(results[k]["sens"][j][1] / DENOM) for k in keys) + r" \\")
    add(r"\bottomrule\end{tabular}\end{table}}")

    with open(out_path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(lines) + "\n")


# =====================================================================
# 7) MAIN EXECUTION
# =====================================================================
def main():
    log_rets, close_prices, divs = fetch_tlt_data()
    dgs1, dgs2 = fetch_fred_yields()

    results = {}
    for k, t0 in VALUATION_DATES:
        ts = pd.Timestamp(t0)

        # Market inputs as of valuation date t0
        S = float(close_prices[close_prices.index <= ts].iloc[-1])
        div_12m = float(divs[(divs.index > ts - pd.Timedelta(days=365)) & (divs.index <= ts)].sum())
        q = float(np.log(1.0 + div_12m / S))
        r, y1, y2 = get_continuous_rf_rate(t0, dgs1, dgs2)
        n_days = int(round(252.0 * year_frac(t0, FINAL_FIX_DATE)))

        # Volatility estimation on trailing 1,260 daily log-returns
        u_win = log_rets[log_rets.index <= ts].iloc[-EST_WINDOW:].values
        g = fit_garch_mle(u_win)
        sig_hist = vol_historical(u_win)
        sig_n = float(np.sqrt(252.0 * g["s2n"]))
        sig_vl = float(np.sqrt(252.0 * g["vl"]))
        sig_garch = garch_option_vol(g["alpha"], g["beta"], g["vl"], g["s2n"], n_days)

        # Closed-form non-callable value & recombining binomial tree valuation
        nc_res = non_callable_value(S, r, q, sig_garch, t0)
        tree_nc, _ = binomial_tree_valuation(S, r, q, sig_garch, t0, steps_per_day=2, callable_flag=False)
        tree_call, called_dates = binomial_tree_valuation(S, r, q, sig_garch, t0, steps_per_day=2, callable_flag=True)
        issuer_call_val = tree_nc - tree_call

        # Sensitivity scenarios
        sens = [
            ("Base case (GARCH(1,1))", tree_call),
            ("Historical 1Y volatility", binomial_tree_valuation(S, r, q, sig_hist, t0, 2, True)[0]),
            ("Volatility $+2$ pp", binomial_tree_valuation(S, r, q, sig_garch + 0.02, t0, 2, True)[0]),
            ("Volatility $-2$ pp", binomial_tree_valuation(S, r, q, sig_garch - 0.02, t0, 2, True)[0]),
            ("No dividends ($q = 0$)", binomial_tree_valuation(S, r, 0.0, sig_garch, t0, 2, True)[0]),
            ("Risk-free rate $r + 0.50$ pp", binomial_tree_valuation(S, r + 0.005, q, sig_garch, t0, 2, True)[0]),
            ("Risk-free rate $r - 0.50$ pp", binomial_tree_valuation(S, r - 0.005, q, sig_garch, t0, 2, True)[0]),
            ("Without issuer call ($V^{\\rm NC}$)", nc_res["nc"]),
        ]

        quote = LEONTEQ_QUOTES[k]
        results[k] = dict(
            val_date=t0, S=S, r=r, y1=y1, y2=y2, q=q, N=n_days, g=g,
            sig_hist=sig_hist, sig_n=sig_n, sig_vl=sig_vl, sig_garch=sig_garch,
            nc_res=nc_res, tree_nc=tree_nc, tree_call=tree_call,
            issuer_call=issuer_call_val, called_dates=called_dates,
            quote=quote, sens=sens,
        )

        print(f"{k} {t0}: S={S:.2f} r={r*100:.2f}% q={q*100:.2f}% | "
              f"GARCH vol={sig_garch*100:.2f}% | BS NC={nc_res['nc']:.2f} | "
              f"Tree NC={tree_nc:.2f} | Tree Callable={tree_call:.2f} ({tree_call/10:.2f}%) | "
              f"Issuer Call={issuer_call_val:.2f} | Active call dates={called_dates} | Leonteq={quote:.2f}%")

    write_latex(OUTPUT_TEX, results)
    print(f"Done -> {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
