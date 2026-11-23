#!/usr/bin/env python3  # Execute script using the Python 3 interpreter
# -*- coding: utf-8 -*-  # Declare UTF-8 source file encoding
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
"""  # Module docstring summarizing the project scope, methodology, and data sources

from datetime import date  # Import date class for exact calendar-day calculations

import numpy as np  # Import NumPy for vectorized numerical arrays and mathematical functions
import pandas as pd  # Import pandas for time-series indexing and CSV reading
import yfinance as yf  # Import yfinance to download historical TLT prices and dividends
from scipy.optimize import minimize  # Import Nelder-Mead optimizer for GARCH(1,1) MLE
from scipy.stats import norm  # Import standard normal distribution for CDF N(x)

# =====================================================================
# 1) CONTRACT PARAMETERS FROM TERMSHEET (ISIN CH1506044580)
# =====================================================================
DENOM = 1000.0  # Certificate denomination / principal amount in USD (1,000 USD)
INIT_FIX = 90.24  # Initial Fixing Level and maturity-only Barrier Level H (90.24 USD)
BONUS_LVL = 1.086  # Bonus Level as a decimal multiplier (108.60%)
STRIKE_BONUS = INIT_FIX * BONUS_LVL  # Effective strike for 200% upside participation (98.00064 USD)
PARTICIPATION = 2.0  # Upside participation rate above the Bonus Level (200%)
EARLY_REDEMP_AMT = 1086.0  # Fixed cash payout upon issuer early redemption (1,000 + 86 USD)
BONUS_COUPON = EARLY_REDEMP_AMT - DENOM  # Binary bonus coupon amount (86.00 USD)
CALL_UNITS = PARTICIPATION * DENOM / INIT_FIX  # Number of European call units (2000 / 90.24 = 22.16312057)
FINAL_FIX_DATE = date(2027, 11, 26)  # Final Fixing Date on which TLT close S_T is observed
REDEMP_DATE = date(2027, 12, 3)  # Final Redemption Date on which maturity cash settlement is paid

OBS_SCHEDULE = [  # List of all 23 (Early Redemption Observation Date, Early Redemption Date) pairs from termsheet
    (date(2025, 12, 26), date(2026, 1, 6)),    # Call date 1: observation 26 Dec 2025, payment 6 Jan 2026
    (date(2026, 1, 26), date(2026, 1, 30)),    # Call date 2: observation 26 Jan 2026, payment 30 Jan 2026
    (date(2026, 2, 25), date(2026, 3, 4)),     # Call date 3: observation 25 Feb 2026, payment 4 Mar 2026
    (date(2026, 3, 25), date(2026, 4, 1)),     # Call date 4: observation 25 Mar 2026, payment 1 Apr 2026
    (date(2026, 4, 27), date(2026, 5, 4)),     # Call date 5: observation 27 Apr 2026, payment 4 May 2026
    (date(2026, 5, 26), date(2026, 6, 1)),     # Call date 6: observation 26 May 2026, payment 1 Jun 2026
    (date(2026, 6, 25), date(2026, 7, 2)),     # Call date 7: observation 25 Jun 2026, payment 2 Jul 2026
    (date(2026, 7, 27), date(2026, 7, 31)),    # Call date 8: observation 27 Jul 2026, payment 31 Jul 2026
    (date(2026, 8, 25), date(2026, 9, 1)),     # Call date 9: observation 25 Aug 2026, payment 1 Sep 2026
    (date(2026, 9, 25), date(2026, 10, 2)),    # Call date 10: observation 25 Sep 2026, payment 2 Oct 2026
    (date(2026, 10, 26), date(2026, 10, 30)),  # Call date 11: observation 26 Oct 2026, payment 30 Oct 2026
    (date(2026, 11, 25), date(2026, 12, 2)),   # Call date 12: observation 25 Nov 2026, payment 2 Dec 2026
    (date(2026, 12, 28), date(2027, 1, 4)),    # Call date 13: observation 28 Dec 2026, payment 4 Jan 2027
    (date(2027, 1, 25), date(2027, 2, 1)),     # Call date 14: observation 25 Jan 2027, payment 1 Feb 2027
    (date(2027, 2, 25), date(2027, 3, 4)),     # Call date 15: observation 25 Feb 2027, payment 4 Mar 2027
    (date(2027, 3, 25), date(2027, 4, 5)),     # Call date 16: observation 25 Mar 2027, payment 5 Apr 2027
    (date(2027, 4, 26), date(2027, 4, 30)),    # Call date 17: observation 26 Apr 2027, payment 30 Apr 2027
    (date(2027, 5, 25), date(2027, 6, 1)),     # Call date 18: observation 25 May 2027, payment 1 Jun 2027
    (date(2027, 6, 25), date(2027, 7, 2)),     # Call date 19: observation 25 Jun 2027, payment 2 Jul 2027
    (date(2027, 7, 26), date(2027, 7, 30)),    # Call date 20: observation 26 Jul 2027, payment 30 Jul 2027
    (date(2027, 8, 25), date(2027, 9, 1)),     # Call date 21: observation 25 Aug 2027, payment 1 Sep 2027
    (date(2027, 9, 27), date(2027, 10, 1)),    # Call date 22: observation 27 Sep 2027, payment 1 Oct 2027
    (date(2027, 10, 25), date(2027, 11, 1)),   # Call date 23: final observation 25 Oct 2027, payment 1 Nov 2027
]  # End of early redemption schedule list

# =====================================================================
# 2) VALUATION DATES & OBSERVED LEONTEQ QUOTES
# =====================================================================
VALUATION_DATES = [  # List of (scenario_key, valuation_date) tuples to evaluate
    ("A", date(2025, 12, 3)),   # Scenario A: Issue Date (3 Dec 2025)
    ("B", date(2026, 10, 23)),  # Scenario B: Required Valuation Date (23 Oct 2026)
    ("C", date(2026, 11, 20)),  # Scenario C: Current Valuation Date (20 Nov 2026)
]  # End of valuation dates list

LEONTEQ_QUOTES = {  # Dictionary of indicative secondary-market quotes from Leonteq (% of notional)
    "A": 98.62,  # Indicative quote on 3 Dec 2025
    "B": 93.97,  # Indicative quote on 23 Oct 2026 (12:00 CET)
    "C": 93.79,  # Indicative quote on 20 Nov 2026 (12:00 CET)
}  # End of Leonteq quotes dictionary

EST_WINDOW = 1260  # Estimation window length in trading days for GARCH(1,1) MLE (approx. 5 years)
OUTPUT_TEX = "outputs.tex"  # Destination filename for exported LaTeX macros and tables


# =====================================================================
# 3) VALUATION MODELS (BLACK-SCHOLES / BLACK + BINOMIAL TREE)
# =====================================================================
def year_frac(d1, d2):  # Compute ACT/365 year fraction between calendar dates d1 and d2
    """ACT/365 year fraction between calendar dates d1 and d2."""  # Function docstring
    return (d2 - d1).days / 365.0  # Divide exact integer calendar-day difference by 365.0


def non_callable_value(S, r, q, sig, t0):  # Compute closed-form risk-free value of non-callable payoff (Eq. 4)
    """Closed-form risk-free valuation of the non-callable payoff (Eq. 4)."""  # Function docstring
    tf = year_frac(t0, FINAL_FIX_DATE)  # Time in years from t0 to Final Fixing Date (option expiry)
    tp = year_frac(t0, REDEMP_DATE)  # Time in years from t0 to Final Redemption Date (cash payment)
    df = np.exp(-r * tp)  # Risk-free discount factor e^{-r * tau_pay} from payment date to t0
    fwd = S * np.exp((r - q) * tf)  # Forward TLT price F_0 = S_t * exp((r - q) * tau_fix)
    vol_t = sig * np.sqrt(tf)  # Total log-return standard deviation sigma * sqrt(tau_fix)

    d2_bar = (np.log(fwd / INIT_FIX) - 0.5 * vol_t ** 2) / vol_t  # Standardized d2 for binary call at barrier H = 90.24
    d1_call = (np.log(fwd / STRIKE_BONUS) + 0.5 * vol_t ** 2) / vol_t  # Standardized d1 for European call at K_b = 98.00064
    d2_call = d1_call - vol_t  # Standardized d2 = d1 - sigma * sqrt(tau_fix) for European call

    bond = DENOM * df  # Present value of zero-coupon bond paying 1,000 USD at maturity
    digital = BONUS_COUPON * df * norm.cdf(d2_bar)  # Present value of USD 86 binary call (Part 2, slide 36)
    call = CALL_UNITS * df * (fwd * norm.cdf(d1_call) - STRIKE_BONUS * norm.cdf(d2_call))  # Present value of 22.163 calls (Black's formula)

    return dict(  # Return dictionary containing forward, bonus probability, and component PVs
        fwd=float(fwd),  # Forward price F_0 in USD
        prob_bonus=float(norm.cdf(d2_bar)),  # Risk-neutral probability Q(S_T > 90.24)
        bond=float(bond),  # Zero-coupon bond value in USD
        digital=float(digital),  # Binary call value in USD
        call=float(call),  # Outperformance European calls value in USD
        nc=float(bond + digital + call),  # Total non-callable certificate value V^NC in USD
    )  # End of return dictionary


def binomial_tree_valuation(S, r, q, sig, t0, steps_per_day=2, callable_flag=True):  # Price certificate on a recombining binomial tree
    """Recombining binomial tree valuation for the 23-date Bermudan callable certificate."""  # Function docstring
    n_steps = (FINAL_FIX_DATE - t0).days * steps_per_day  # Total number of tree steps (2 steps per calendar day)
    dt = 1.0 / (365.0 * steps_per_day)  # Duration of a single tree step in years
    u = np.exp(sig * np.sqrt(dt))  # Upward price factor u = exp(sigma * sqrt(dt))
    d = 1.0 / u  # Downward price factor d = 1 / u ensuring tree recombination
    p = (np.exp((r - q) * dt) - d) / (u - d)  # Risk-neutral up-move probability adjusted for continuous yield q
    disc = np.exp(-r * dt)  # Single-step risk-free discount factor e^{-r * dt}

    i = np.arange(n_steps + 1)  # Array of downward move counts (0 to n_steps) at maturity step
    S_T = S * (u ** (n_steps - 2 * i))  # Terminal TLT prices across all nodes on Final Fixing Date
    payoff_T = DENOM + BONUS_COUPON * (S_T > INIT_FIX) + CALL_UNITS * np.maximum(S_T - STRIKE_BONUS, 0.0)  # Exact terminal payoff (Eq. 2)
    V = np.exp(-r * year_frac(FINAL_FIX_DATE, REDEMP_DATE)) * payoff_T  # Discount terminal payoff 7 days from REDEMP_DATE to FINAL_FIX_DATE

    obs_map = {  # Dictionary mapping tree step indices of future call dates to (obs_number, discounted 1,086 USD)
        (o - t0).days * steps_per_day: (k + 1, EARLY_REDEMP_AMT * np.exp(-r * year_frac(o, pay)))  # Step index -> (k+1, call payout PV)
        for k, (o, pay) in enumerate(OBS_SCHEDULE) if o > t0  # Include only call observation dates strictly after t0
    }  # End of observation date mapping

    called_dates = []  # List recording observation date numbers where issuer strictly prefers calling
    for j in range(n_steps - 1, -1, -1):  # Backward induction loop from step n_steps - 1 down to root j = 0
        V = disc * (p * V[:j + 1] + (1.0 - p) * V[1:j + 2])  # Discounted expected continuation value at step j
        if callable_flag and j in obs_map:  # Check if early call is active and step j is an observation date
            k_num, call_pv = obs_map[j]  # Retrieve observation index k_num and present value of USD 1,086 call payout
            if np.any(call_pv < V):  # Check if calling is strictly cheaper than continuation at any node
                called_dates.append(k_num)  # Record observation index k_num as an active exercise date
            V = np.minimum(V, call_pv)  # Issuer minimizes liability: take min(continuation value, call_pv)

    called_dates.sort()  # Sort active exercise observation indices in ascending chronological order
    return float(V[0]), called_dates  # Return root certificate value at t0 (USD) and list of active call dates


# =====================================================================
# 4) VOLATILITY ESTIMATION (HISTORICAL & GARCH(1,1) MLE)
# =====================================================================
def vol_historical(u, window=252):  # Compute 1-year (252-day) equally weighted annualized historical volatility
    """1-year (252-day) equally weighted annualized historical volatility."""  # Function docstring
    return float(np.std(u[-window:], ddof=1) * np.sqrt(252.0))  # Sample standard deviation scaled by sqrt(252)


def fit_garch_mle(u):  # Calibrate GARCH(1,1) parameters via Gaussian Maximum Likelihood (Part 1, slide 51)
    """Estimate GARCH(1,1) parameters by maximizing Gaussian log-likelihood (Part 1, slide 51)."""  # Function docstring
    x = np.asarray(u, float) * 100.0  # Scale daily log-returns to percentages for better numerical conditioning
    v0 = float(np.var(x))  # Compute sample variance in percent squared to initialize variance recursion

    def neg_log_lik(params):  # Objective function equal to sum[ ln(sigma_i^2) + u_i^2 / sigma_i^2 ]
        omega, alpha, beta = params  # Unpack candidate GARCH(1,1) parameters (omega, alpha, beta)
        if omega <= 1e-8 or alpha <= 0.0 or beta <= 0.0 or (alpha + beta) >= 0.9995:  # Check positivity and stationarity
            return 1e12  # Return large penalty if parameters violate GARCH(1,1) constraints
        s2 = v0  # Initialize conditional daily variance at unconditional sample variance
        total = 0.0  # Initialize negative log-likelihood accumulator
        for xi in x:  # Loop over all percentage log-returns in the estimation window
            total += np.log(s2) + (xi * xi) / s2  # Accumulate day-i Gaussian negative log-likelihood term
            s2 = omega + alpha * (xi * xi) + beta * s2  # Update conditional variance for next trading day (Eq. 5)
        return total  # Return total negative log-likelihood value

    best = None  # Variable storing the best optimization result across multiple starting guesses
    for a0, b0 in [(0.05, 0.90), (0.03, 0.95), (0.08, 0.88)]:  # Loop over three standard initial (alpha, beta) pairs
        w0 = v0 * (1.0 - a0 - b0)  # Compute initial intercept omega_0 = (1 - alpha_0 - beta_0) * sample_var
        res = minimize(  # Run derivative-free Nelder-Mead optimization
            neg_log_lik, [w0, a0, b0], method="Nelder-Mead",  # Pass objective, initial vector, and method
            options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-7},  # Specify iteration cap and convergence tolerances
        )  # End of minimize call
        if best is None or res.fun < best.fun:  # Keep candidate achieving the lowest negative log-likelihood
            best = res  # Update best fit

    w_pct, a_hat, b_hat = best.x  # Extract optimal parameter estimates from best solution
    w_d = w_pct / 1e4  # Convert intercept omega back from percent squared to decimal units
    vl_d = w_d / (1.0 - a_hat - b_hat)  # Compute long-run daily variance V_L = omega / (1 - alpha - beta)

    s2 = v0 / 1e4  # Initialize decimal conditional variance at start of sample
    for xi in u:  # Filter historical daily log-returns forward to the valuation date
        s2 = w_d + a_hat * (xi * xi) + b_hat * s2  # Update conditional daily variance sigma_n^2
    return dict(omega=w_d, alpha=a_hat, beta=b_hat, vl=vl_d, s2n=s2, n=len(u))  # Return calibrated parameter dictionary


def garch_option_vol(alpha, beta, vl, s2n, n_days):  # Compute expected annualized volatility over option life (Part 1, slides 52-53)
    """Expected annualized volatility over remaining option life of n_days (Part 1, slides 52-53)."""  # Function docstring
    k = np.arange(n_days)  # Create array of horizon day indices k = 0, 1, ..., N - 1
    expected_vars = vl + ((alpha + beta) ** k) * (s2n - vl)  # Multi-step variance forecast E[sigma_{n+k}^2] (Part 1, slide 52)
    return float(np.sqrt(252.0 * np.mean(expected_vars)))  # Annualize average expected daily variance over N days (Eq. 6)


# =====================================================================
# 5) LIVE MARKET DATA DOWNLOAD (YAHOO FINANCE + FRED)
# =====================================================================
def fetch_tlt_data():  # Download daily TLT prices, adjusted log-returns, and dividends from Yahoo Finance
    """Download TLT daily prices, dividend-adjusted log-returns, and cash dividends from Yahoo Finance."""  # Function docstring
    df = yf.download("TLT", start="2015-01-01", auto_adjust=False, progress=False)  # Download daily TLT OHLCV data since 2015
    if hasattr(df.columns, "droplevel") and df.columns.nlevels > 1:  # Check if yfinance returned MultiIndex columns
        df.columns = df.columns.droplevel(1)  # Drop ticker level so column names are flat strings
    if getattr(df.index, "tz", None) is not None:  # Check if DatetimeIndex carries timezone metadata
        df.index = df.index.tz_localize(None)  # Convert index to timezone-naive timestamps for clean comparisons
    df = df.sort_index()  # Ensure chronological ascending date order

    adj = df["Adj Close"].squeeze().dropna()  # Extract dividend-adjusted close series for return calculations
    close = df["Close"].squeeze().dropna()  # Extract raw closing price series for spot price lookups
    log_rets = np.log(adj / adj.shift(1)).dropna()  # Compute daily log-returns u_i = ln(S_i / S_{i-1})

    divs = yf.Ticker("TLT").dividends  # Download historical monthly cash dividend payouts of TLT
    if getattr(divs.index, "tz", None) is not None:  # Check if dividend series index has timezone metadata
        divs.index = divs.index.tz_localize(None)  # Strip timezone information from dividend dates
    divs = divs.sort_index()  # Sort dividend payments chronologically
    return log_rets, close, divs  # Return tuple of (log_returns, close_prices, dividends)


def fetch_fred_yields():  # Download daily 1-Year (DGS1) and 2-Year (DGS2) U.S. Treasury yields from FRED
    """Download daily 1-Year (DGS1) and 2-Year (DGS2) U.S. Treasury yields from FRED."""  # Function docstring
    d1 = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1", index_col=0, parse_dates=True)  # Read 1Y CMT CSV
    d2 = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2", index_col=0, parse_dates=True)  # Read 2Y CMT CSV
    s1 = pd.to_numeric(d1.iloc[:, 0], errors="coerce").dropna().sort_index() / 100.0  # Convert 1Y yield strings to decimals
    s2 = pd.to_numeric(d2.iloc[:, 0], errors="coerce").dropna().sort_index() / 100.0  # Convert 2Y yield strings to decimals
    return s1, s2  # Return tuple of (1Y_yield_series, 2Y_yield_series)


def get_continuous_rf_rate(t0, dgs1, dgs2):  # Interpolate 1Y/2Y yields to tau_pay and convert to continuous compounding
    """Linearly interpolate 1Y/2Y Treasury yields to tau_pay and convert to continuous compounding."""  # Function docstring
    ts = pd.Timestamp(t0)  # Convert valuation date t0 to pandas Timestamp
    y1 = float(dgs1[dgs1.index <= ts].iloc[-1])  # Look up latest available 1Y Treasury yield on or before t0
    y2 = float(dgs2[dgs2.index <= ts].iloc[-1])  # Look up latest available 2Y Treasury yield on or before t0
    tp = year_frac(t0, REDEMP_DATE)  # Compute remaining time to payment date in years (between 1 and 2 years)
    w = float(np.clip(tp - 1.0, 0.0, 1.0))  # Compute linear interpolation weight on the 2Y Treasury yield
    y_par = (1.0 - w) * y1 + w * y2  # Compute linearly interpolated semi-annual bond-equivalent yield
    r_cont = float(2.0 * np.log(1.0 + y_par / 2.0))  # Convert semi-annual yield to continuous rate r = 2*ln(1 + y/2)
    return r_cont, y1, y2  # Return continuous risk-free rate r along with raw 1Y and 2Y yields


# =====================================================================
# 6) LATEX EXPORT HELPERS
# =====================================================================
def fmt_num(x, nd=2):  # Format float x to nd decimal places for LaTeX output
    if x < 0:  # Check if x is negative
        return "$-$" + fmt_num(-x, nd)  # Prepend proper LaTeX math minus sign and format magnitude
    return f"{x:,.{nd}f}".replace(",", r"\,")  # Format with thin space \, as thousands separator


def fmt_pct(x, nd=2):  # Format decimal value x as a LaTeX percentage string
    return fmt_num(100.0 * x, nd) + r"\,\%"  # Multiply by 100 and append thin space + escaped percent sign


def fmt_date(d):  # Format datetime.date object as 'D Mon YYYY'
    return d.strftime("%d %b %Y").lstrip("0")  # Strip leading zero from single-digit days


def fmt_sci(x):  # Format very small floats in LaTeX scientific notation
    m, e = f"{x:.3e}".split("e")  # Split scientific notation string into mantissa and exponent
    return f"{m}\\cdot 10^{{{int(e)}}}"  # Return LaTeX math expression m \cdot 10^{e}


def write_latex(out_path, results):  # Generate outputs.tex containing all scalar macros and tables for main.tex
    """Write all scalar macros and LaTeX tables consumed by main.tex."""  # Function docstring
    lines = []  # Initialize list storing output LaTeX lines
    add = lines.append  # Bind local shorthand for appending lines
    keys = list(results)  # Extract ordered scenario keys (["A", "B", "C"])

    def macro(name, body):  # Helper appending a LaTeX \newcommand definition
        add(f"\\newcommand{{\\{name}}}{{{body}}}")  # Format and append \newcommand{\name}{body}

    for k in keys:  # Loop over scenarios A, B, C to export scalar macros used in text
        i = results[k]  # Retrieve scenario results dictionary
        nc = i["nc_res"]  # Retrieve non-callable valuation dictionary
        m_pct = i["tree_call"] / 10.0  # Convert callable certificate value (USD) to % of notional
        qt = i["quote"]  # Retrieve observed Leonteq quote in % of notional
        macro(f"date{k}", fmt_date(i["val_date"]))  # Export formatted valuation date
        macro(f"S{k}", fmt_num(i["S"]))  # Export TLT spot price
        macro(f"r{k}", fmt_pct(i["r"]))  # Export continuous risk-free rate r
        macro(f"q{k}", fmt_pct(i["q"]))  # Export continuous dividend yield q
        macro(f"sigg{k}", fmt_pct(i["sig_garch"]))  # Export expected GARCH(1,1) volatility over option life
        macro(f"sign{k}", fmt_pct(i["sig_n"]))  # Export next-day GARCH spot volatility
        macro(f"sigvl{k}", fmt_pct(i["sig_vl"]))  # Export long-run GARCH volatility
        macro(f"nc{k}", fmt_num(nc["nc"]))  # Export non-callable value in USD
        macro(f"ncp{k}", fmt_pct(nc["nc"] / DENOM))  # Export non-callable value in % of notional
        macro(f"call{k}", fmt_num(i["issuer_call"]))  # Export issuer Bermudan call value in USD
        macro(f"c{k}", fmt_pct(i["tree_call"] / DENOM))  # Export callable certificate value in % of notional
        macro(f"cusd{k}", fmt_num(i["tree_call"]))  # Export callable certificate value in USD
        macro(f"qt{k}", fmt_pct(qt / 100.0))  # Export Leonteq quote in % of notional
        macro(f"gap{k}", fmt_num(100.0 - m_pct))  # Export gap between 100% issue price and model value (pp)
        macro(f"diff{k}", fmt_num(m_pct - qt))  # Export Model minus Leonteq difference (pp)
        macro(f"exobs{k}", ", ".join(str(x) for x in i["called_dates"]) or "none")  # Export active call date index

    col_spec = "c" * len(keys)  # Build centered column specification string ("ccc")
    header = " & ".join(fmt_date(results[k]["val_date"]) for k in keys)  # Build table column header row of dates

    def row(label, fn):  # Helper formatting one table row across all three valuation dates
        return f"{label} & " + " & ".join(fn(results[k]) for k in keys) + r" \\"  # Join columns with & and end with \\

    def fmt_call_dates(cd):  # Format list of active exercise observation numbers into human-readable dates
        if not cd:  # Handle case where issuer never exercises early
            return "none"  # Return "none" string
        return ", ".join(f"{fmt_date(OBS_SCHEDULE[idx - 1][0])} (no.\\ {idx})" for idx in cd)  # Format date and index

    add(r"\newcommand{\TabInputs}{\begin{table}[ht]\centering\small")  # Start Table 2 (\TabInputs) macro
    add(r"\caption{Market inputs (Yahoo Finance, FRED) and volatility estimates.}\label{tab:inputs}")  # Caption and label
    add(f"\\begin{{tabular}}{{@{{}}l{col_spec}@{{}}}}\\toprule")  # Open tabular environment with top rule
    add(f" & {header} \\\\\\midrule")  # Write valuation dates header and midrule
    add(row("TLT close price $S_t$ (USD)", lambda i: fmt_num(i["S"])))  # Row: TLT spot closing price
    add(row(r"Time to final fixing $\tau_{\rm fix}$ (yr) / $N$ days",  # Row: time to fixing in years and trading days
            lambda i: f"{fmt_num(year_frac(i['val_date'], FINAL_FIX_DATE), 3)} / {i['N']}"))  # Format tau_fix / N
    add(row(r"Time to payment $\tau_{\rm pay}$ (yr)", lambda i: fmt_num(year_frac(i["val_date"], REDEMP_DATE), 3)))  # Row: tau_pay
    add(row("FRED Treasury 1Y / 2Y yield", lambda i: f"{fmt_pct(i['y1'])} / {fmt_pct(i['y2'])}"))  # Row: raw FRED 1Y/2Y yields
    add(row("Risk-free rate $r$ (continuous)", lambda i: fmt_pct(i["r"])))  # Row: continuous risk-free rate r
    add(row("Dividend yield $q$ (trailing 12m)", lambda i: fmt_pct(i["q"])))  # Row: continuous trailing dividend yield q
    add(r"\midrule")  # Midrule separating market inputs from volatility estimates
    add(row("GARCH(1,1) expected volatility (over $N$ days)", lambda i: fmt_pct(i["sig_garch"])))  # Row: GARCH expected vol
    add(row("Historical volatility (last 252 days)", lambda i: fmt_pct(i["sig_hist"])))  # Row: 1-year historical vol
    add(r"\bottomrule\end{tabular}\end{table}}")  # Close Table 2 (\TabInputs) macro

    add(r"\newcommand{\TabGarch}{\begin{table}[ht]\centering\small")  # Start Table 3 (\TabGarch) macro
    add(r"\caption{GARCH(1,1) parameters estimated by Maximum Likelihood (1,260 daily log-returns).}\label{tab:garch}")  # Caption
    add(f"\\begin{{tabular}}{{@{{}}l{col_spec}@{{}}}}\\toprule")  # Open tabular environment
    add(f" & {header} \\\\\\midrule")  # Date column headers
    add(row(r"$\omega$", lambda i: "$" + fmt_sci(i["g"]["omega"]) + "$"))  # Row: GARCH intercept omega
    add(row(r"$\alpha$", lambda i: fmt_num(i["g"]["alpha"], 4)))  # Row: ARCH parameter alpha
    add(row(r"$\beta$", lambda i: fmt_num(i["g"]["beta"], 4)))  # Row: GARCH persistence parameter beta
    add(row(r"$\gamma = 1 - \alpha - \beta$", lambda i: fmt_num(1.0 - i["g"]["alpha"] - i["g"]["beta"], 4)))  # Row: gamma
    add(row(r"Long-run volatility $\sqrt{252\,V_L}$", lambda i: fmt_pct(i["sig_vl"])))  # Row: annualized long-run vol
    add(row(r"Next-day volatility $\sigma_n$", lambda i: fmt_pct(i["sig_n"])))  # Row: annualized next-day spot vol
    add(row(r"Expected volatility over option life", lambda i: fmt_pct(i["sig_garch"])))  # Row: average expected vol
    add(r"\bottomrule\end{tabular}\end{table}}")  # Close Table 3 (\TabGarch) macro

    add(r"\newcommand{\TabNonCallable}{\begin{table}[ht]\centering\small")  # Start Table 4 (\TabNonCallable) macro
    add(r"\caption{Valuation without the issuer's call option (USD per 1,000 USD certificate).}\label{tab:nc}")  # Caption
    add(f"\\begin{{tabular}}{{@{{}}l{col_spec}@{{}}}}\\toprule")  # Open tabular environment
    add(f" & {header} \\\\\\midrule")  # Date column headers
    add(row("Forward TLT price $F_0$ (USD)", lambda i: fmt_num(i["nc_res"]["fwd"])))  # Row: forward price F_0
    add(row(r"Risk-neutral prob.\ $\mathbb{Q}(S_T > 90.24)$", lambda i: fmt_pct(i["nc_res"]["prob_bonus"], 1)))  # Row: bonus prob
    add(r"\midrule")  # Midrule separating inputs from valuation components
    add(row("Zero-coupon bond (USD 1,000)", lambda i: fmt_num(i["nc_res"]["bond"])))  # Row: zero-coupon bond PV
    add(row("Binary call (USD 86, $K = 90.24$)", lambda i: fmt_num(i["nc_res"]["digital"])))  # Row: binary call PV
    add(row("22.163 European calls ($K_b = 98.00$)", lambda i: fmt_num(i["nc_res"]["call"])))  # Row: European calls PV
    add(r"\midrule")  # Midrule before non-callable total
    add(row(r"Total non-callable value $V^{\rm NC}$ (USD)", lambda i: fmt_num(i["nc_res"]["nc"])))  # Row: total in USD
    add(row(r"Total non-callable value (\% of notional)", lambda i: fmt_pct(i["nc_res"]["nc"] / DENOM)))  # Row: total in %
    add(r"\bottomrule\end{tabular}\end{table}}")  # Close Table 4 (\TabNonCallable) macro

    add(r"\newcommand{\TabTree}{\begin{table}[ht]\centering\small")  # Start Table 5 (\TabTree) macro
    add(r"\caption{Binomial tree valuation (2 steps/day) and comparison with Leonteq quotes.}\label{tab:tree}")  # Caption
    add(f"\\begin{{tabular}}{{@{{}}l{col_spec}@{{}}}}\\toprule")  # Open tabular environment
    add(f" & {header} \\\\\\midrule")  # Date column headers
    add(row(r"Closed-form non-callable $V^{\rm NC}$ (USD)", lambda i: fmt_num(i["nc_res"]["nc"])))  # Row: closed-form benchmark
    add(row("Binomial tree without issuer call (USD)", lambda i: fmt_num(i["tree_nc"])))  # Row: non-callable tree check
    add(row("Binomial tree with issuer call (USD)", lambda i: fmt_num(i["tree_call"])))  # Row: full callable certificate PV
    add(row("Value of issuer's Bermudan call (USD)", lambda i: fmt_num(i["issuer_call"])))  # Row: issuer call value
    add(row("Call date(s) with optimal exercise", lambda i: fmt_call_dates(i["called_dates"])))  # Row: active exercise date(s)
    add(r"\midrule")  # Midrule before market comparison
    add(row(r"\textbf{Model certificate value (\% notional)}", lambda i: r"\textbf{" + fmt_pct(i["tree_call"] / DENOM) + "}"))  # Bold %
    add(row(r"Leonteq quote (\% notional)", lambda i: fmt_pct(i["quote"] / 100.0)))  # Row: Leonteq quote
    add(row("Difference: Model $-$ Leonteq (pp)", lambda i: fmt_num(i["tree_call"] / 10.0 - i["quote"])))  # Row: Model - Quote
    add(r"\bottomrule\end{tabular}\end{table}}")  # Close Table 5 (\TabTree) macro

    add(r"\newcommand{\TabSens}{\begin{table}[ht]\centering\small")  # Start Table 6 (\TabSens) macro
    add(r"\caption{Sensitivity of the callable certificate value (\% of notional).}\label{tab:sens}")  # Caption
    add(f"\\begin{{tabular}}{{@{{}}l{col_spec}@{{}}}}\\toprule")  # Open tabular environment
    add(f" & {header} \\\\\\midrule")  # Date column headers
    for j in range(len(results[keys[0]]["sens"])):  # Loop over all 8 sensitivity scenarios
        lbl = results[keys[0]]["sens"][j][0]  # Extract scenario label
        add(f"{lbl} & " + " & ".join(fmt_pct(results[k]["sens"][j][1] / DENOM) for k in keys) + r" \\")  # Write scenario row
    add(r"\bottomrule\end{tabular}\end{table}}")  # Close Table 6 (\TabSens) macro

    with open(out_path, "w", encoding="utf-8") as f_out:  # Open outputs.tex in UTF-8 write mode
        f_out.write("\n".join(lines) + "\n")  # Join all lines with newline characters and write to disk


# =====================================================================
# 7) MAIN EXECUTION
# =====================================================================
def main():  # Main workflow coordinating data fetch, GARCH calibration, valuation, and LaTeX export
    log_rets, close_prices, divs = fetch_tlt_data()  # Download TLT log-returns, closing prices, and cash dividends
    dgs1, dgs2 = fetch_fred_yields()  # Download FRED 1-Year and 2-Year U.S. Treasury yields

    results = {}  # Initialize dictionary storing results for scenarios A, B, and C
    for k, t0 in VALUATION_DATES:  # Iterate over all three valuation dates
        ts = pd.Timestamp(t0)  # Convert valuation date to pandas Timestamp

        S = float(close_prices[close_prices.index <= ts].iloc[-1])  # Look up TLT closing price on or before t0
        div_12m = float(divs[(divs.index > ts - pd.Timedelta(days=365)) & (divs.index <= ts)].sum())  # Sum trailing 12m dividends
        q = float(np.log(1.0 + div_12m / S))  # Convert trailing 12m dividend yield to continuous yield q
        r, y1, y2 = get_continuous_rf_rate(t0, dgs1, dgs2)  # Compute interpolated continuous risk-free rate r
        n_days = int(round(252.0 * year_frac(t0, FINAL_FIX_DATE)))  # Compute remaining trading days until Final Fixing Date

        u_win = log_rets[log_rets.index <= ts].iloc[-EST_WINDOW:].values  # Slice last 1,260 daily log-returns up to t0
        g = fit_garch_mle(u_win)  # Calibrate GARCH(1,1) parameters via Gaussian MLE
        sig_hist = vol_historical(u_win)  # Compute 252-day historical volatility for comparison
        sig_n = float(np.sqrt(252.0 * g["s2n"]))  # Annualize next-day GARCH spot volatility sigma_n
        sig_vl = float(np.sqrt(252.0 * g["vl"]))  # Annualize long-run GARCH volatility sqrt(252 * V_L)
        sig_garch = garch_option_vol(g["alpha"], g["beta"], g["vl"], g["s2n"], n_days)  # Compute expected vol over N days

        nc_res = non_callable_value(S, r, q, sig_garch, t0)  # Price non-callable certificate via closed-form Eq. 4
        tree_nc, _ = binomial_tree_valuation(S, r, q, sig_garch, t0, steps_per_day=2, callable_flag=False)  # Tree check (no call)
        tree_call, called_dates = binomial_tree_valuation(S, r, q, sig_garch, t0, steps_per_day=2, callable_flag=True)  # Callable tree
        issuer_call_val = tree_nc - tree_call  # Compute value of issuer's Bermudan call option (USD)

        sens = [  # Compute sensitivity table across 8 scenarios (% of notional)
            ("Base case (GARCH(1,1))", tree_call),  # Main callable valuation with GARCH(1,1) expected vol
            ("Historical 1Y volatility", binomial_tree_valuation(S, r, q, sig_hist, t0, 2, True)[0]),  # Using 252d hist vol
            ("Volatility $+2$ pp", binomial_tree_valuation(S, r, q, sig_garch + 0.02, t0, 2, True)[0]),  # +2 pp vol shock
            ("Volatility $-2$ pp", binomial_tree_valuation(S, r, q, sig_garch - 0.02, t0, 2, True)[0]),  # -2 pp vol shock
            ("No dividends ($q = 0$)", binomial_tree_valuation(S, r, 0.0, sig_garch, t0, 2, True)[0]),  # Ignoring dividends
            ("Risk-free rate $r + 0.50$ pp", binomial_tree_valuation(S, r + 0.005, q, sig_garch, t0, 2, True)[0]),  # +50 bps rate
            ("Risk-free rate $r - 0.50$ pp", binomial_tree_valuation(S, r - 0.005, q, sig_garch, t0, 2, True)[0]),  # -50 bps rate
            ("Without issuer call ($V^{\\rm NC}$)", nc_res["nc"]),  # Non-callable closed-form upper bound
        ]  # End of sensitivity scenarios list

        quote = LEONTEQ_QUOTES[k]  # Retrieve observed Leonteq secondary-market quote
        results[k] = dict(  # Store all computed outputs for scenario k
            val_date=t0, S=S, r=r, y1=y1, y2=y2, q=q, N=n_days, g=g,  # Inputs, date, and GARCH parameters
            sig_hist=sig_hist, sig_n=sig_n, sig_vl=sig_vl, sig_garch=sig_garch,  # Volatility estimates
            nc_res=nc_res, tree_nc=tree_nc, tree_call=tree_call,  # Closed-form and tree valuations
            issuer_call=issuer_call_val, called_dates=called_dates,  # Issuer call value and active dates
            quote=quote, sens=sens,  # Market quote and sensitivity results
        )  # End of scenario dictionary

        print(f"{k} {t0}: S={S:.2f} r={r*100:.2f}% q={q*100:.2f}% | "  # Print summary line to console (inputs)
              f"GARCH vol={sig_garch*100:.2f}% | BS NC={nc_res['nc']:.2f} | "  # Print GARCH vol and closed-form V^NC
              f"Tree NC={tree_nc:.2f} | Tree Callable={tree_call:.2f} ({tree_call/10:.2f}%) | "  # Print tree valuations
              f"Issuer Call={issuer_call_val:.2f} | Active call dates={called_dates} | Leonteq={quote:.2f}%")  # Print call & quote

    write_latex(OUTPUT_TEX, results)  # Export all macros and tables to outputs.tex
    print(f"Done -> {OUTPUT_TEX}")  # Print completion confirmation message


if __name__ == "__main__":  # Run main() only when script is executed directly
    main()  # Invoke main valuation workflow
