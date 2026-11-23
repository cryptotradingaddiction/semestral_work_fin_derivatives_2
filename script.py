#!/usr/bin/env python3  # Shebang telling Unix-like shells to execute the script with Python 3
# -*- coding: utf-8 -*-  # Declare UTF-8 source file encoding
"""
1BP451 Financial Derivatives II - Project
Valuation of the Bonus Outperformance Certificate on iShares 20+ Year Treasury Bond ETF (TLT),
ISIN CH1506044580, issued by Banque Internationale a Luxembourg S.A. (BIL).
"""  # Module docstring identifying the course project and financial instrument

import argparse  # Import standard library module for parsing command-line arguments (--data, --out)
from datetime import date  # Import date class for exact calendar-day arithmetic

import numpy as np  # Import NumPy for vectorized numerical arrays and quadrature integration
from scipy.optimize import brentq, minimize  # Import brentq (1D root-finder) and minimize (BFGS optimizer)
from scipy.special import expit  # Import numerically stable logistic sigmoid function 1 / (1 + exp(-x))
from scipy.stats import norm  # Import standard normal distribution for CDF N(x) and PDF phi(x)

# =====================================================================
# 1) CONTRACT PARAMETERS FROM TERMSHEET (ISIN CH1506044580)
# =====================================================================
DENOM = 1000.0  # Certificate denomination in USD (1,000 USD per termsheet)
INIT_FIX = 90.24  # Initial Fixing Level (90.24 USD) and 100% maturity-only barrier level H
BONUS_LVL = 1.086  # Bonus Level as a decimal multiplier (108.60%)
STRIKE_BONUS = INIT_FIX * BONUS_LVL  # Effective strike for 200% upside participation: 90.24 * 1.086 = 98.00064 USD
PARTICIPATION = 2.0  # Upside participation rate above the Bonus Level (200.00%)
EARLY_REDEMP_AMT = 1086.0  # Fixed cash settlement paid if issuer exercises early redemption (1,000 + 86 USD)
BONUS_COUPON = EARLY_REDEMP_AMT - DENOM  # Digital bonus coupon paid above barrier or on early call (86.00 USD)
CALL_UNITS = PARTICIPATION * DENOM / INIT_FIX  # Number of European call units in exact payoff decomposition (22.16312057)
FINAL_FIX_DATE = date(2027, 11, 26)  # Final Fixing Date on which TLT closing price S_T is observed (26 Nov 2027)
REDEMP_DATE = date(2027, 12, 3)  # Final Redemption Date on which maturity cash settlement is paid (3 Dec 2027)

OBS_SCHEDULE = [  # List of all 23 (Early Redemption Observation Date, Early Redemption Payment Date) pairs
    (date(2025, 12, 26), date(2026, 1, 6)),    # Observation 1 (26 Dec 2025) and settlement (6 Jan 2026)
    (date(2026, 1, 26), date(2026, 1, 30)),    # Observation 2 (26 Jan 2026) and settlement (30 Jan 2026)
    (date(2026, 2, 25), date(2026, 3, 4)),     # Observation 3 (25 Feb 2026) and settlement (4 Mar 2026)
    (date(2026, 3, 25), date(2026, 4, 1)),     # Observation 4 (25 Mar 2026) and settlement (1 Apr 2026)
    (date(2026, 4, 27), date(2026, 5, 4)),     # Observation 5 (27 Apr 2026) and settlement (4 May 2026)
    (date(2026, 5, 26), date(2026, 6, 1)),     # Observation 6 (26 May 2026) and settlement (1 Jun 2026)
    (date(2026, 6, 25), date(2026, 7, 2)),     # Observation 7 (25 Jun 2026) and settlement (2 Jul 2026)
    (date(2026, 7, 27), date(2026, 7, 31)),    # Observation 8 (27 Jul 2026) and settlement (31 Jul 2026)
    (date(2026, 8, 25), date(2026, 9, 1)),     # Observation 9 (25 Aug 2026) and settlement (1 Sep 2026)
    (date(2026, 9, 25), date(2026, 10, 2)),    # Observation 10 (25 Sep 2026) and settlement (2 Oct 2026)
    (date(2026, 10, 26), date(2026, 10, 30)),  # Observation 11 (26 Oct 2026) and settlement (30 Oct 2026)
    (date(2026, 11, 25), date(2026, 12, 2)),   # Observation 12 (25 Nov 2026) and settlement (2 Dec 2026)
    (date(2026, 12, 28), date(2027, 1, 4)),    # Observation 13 (28 Dec 2026) and settlement (4 Jan 2027)
    (date(2027, 1, 25), date(2027, 2, 1)),     # Observation 14 (25 Jan 2027) and settlement (1 Feb 2027)
    (date(2027, 2, 25), date(2027, 3, 4)),     # Observation 15 (25 Feb 2027) and settlement (4 Mar 2027)
    (date(2027, 3, 25), date(2027, 4, 5)),     # Observation 16 (25 Mar 2027) and settlement (5 Apr 2027)
    (date(2027, 4, 26), date(2027, 4, 30)),    # Observation 17 (26 Apr 2027) and settlement (30 Apr 2027)
    (date(2027, 5, 25), date(2027, 6, 1)),     # Observation 18 (25 May 2027) and settlement (1 Jun 2027)
    (date(2027, 6, 25), date(2027, 7, 2)),     # Observation 19 (25 Jun 2027) and settlement (2 Jul 2027)
    (date(2027, 7, 26), date(2027, 7, 30)),    # Observation 20 (26 Jul 2027) and settlement (30 Jul 2027)
    (date(2027, 8, 25), date(2027, 9, 1)),     # Observation 21 (25 Aug 2027) and settlement (1 Sep 2027)
    (date(2027, 9, 27), date(2027, 10, 1)),    # Observation 22 (27 Sep 2027) and settlement (1 Oct 2027)
    (date(2027, 10, 25), date(2027, 11, 1)),   # Observation 23 (25 Oct 2027) and settlement (1 Nov 2027)
]  # End of Bermudan early redemption schedule list
T23, PAY23 = OBS_SCHEDULE[-1]  # Unpack the final (23rd) observation date and its settlement date

# =====================================================================
# 2) MARKET INPUTS FOR VALUATION DATES
# =====================================================================
VALUATIONS = [  # List of valuation scenarios: Issue Date (A), Required Date (B), and Current Date (C)
    dict(key="A", val_date=date(2025, 12, 3), S=89.06, r=0.0415, q=0.0495, s=0.0080),   # Scenario A inputs
    dict(key="B", val_date=date(2026, 10, 23), S=77.98, r=0.0400, q=0.0495, s=0.0080),  # Scenario B inputs
    dict(key="C", val_date=date(2026, 11, 20), S=81.45, r=0.0400, q=0.0487, s=0.0080),  # Scenario C inputs
]  # End of valuation scenarios list

LEONTEQ_QUOTES = {  # Dictionary mapping scenario keys to indicative Leonteq quotes (bid, ask) in % of notional
    "A": (98.62, 98.62),  # Indicative Leonteq quote on 3 Dec 2025
    "B": (93.97, 93.97),  # Indicative Leonteq quote on 23 Oct 2026
    "C": (93.79, 93.79),  # Indicative Leonteq quote on 20 Nov 2026
}  # End of Leonteq quotes dictionary

FALLBACK_GARCH = dict(vl_ann=0.13, alpha=0.06, beta=0.91)  # Fallback GARCH(1,1) parameters if price data fail to load
FALLBACK_SIGN = {"A": 0.12, "B": 0.143, "C": 0.143}  # Fallback spot annualized volatilities per valuation scenario
EST_WINDOW = 1260  # Rolling estimation window in trading days for GARCH(1,1) MLE (approx. 5 years)
EWMA_LAMBDA = 0.97  # Exponential decay factor lambda = 0.97 for EWMA volatility estimation


# =====================================================================
# 3) VALUATION FUNCTIONS (PART 1 & PART 2)
# =====================================================================
def year_frac(d1, d2):  # Define helper computing ACT/365 year fraction between two calendar dates
    return (d2 - d1).days / 365.0  # Divide exact integer calendar days between d1 and d2 by 365.0


def non_callable_parts(S, r, q, s, sig, t0, Tfix=FINAL_FIX_DATE, Tpay=REDEMP_DATE):  # Price the 3 non-callable payoff components at t0
    S = np.asarray(S, float)  # Cast spot price S to float NumPy array so function works for scalars and vectors
    tf, tp = year_frac(t0, Tfix), year_frac(t0, Tpay)  # Compute time to fixing (tf) and time to payment (tp) in years
    df_credit = np.exp(-(r + s) * tp)  # Compute credit-adjusted discount factor e^{-(r+s)*tau_pay} for maturity payout
    fwd = S * np.exp((r - q) * tf)  # Compute risk-neutral forward price F_t = S_t * exp((r - q) * tau_fix)
    vol_sqrt_t = sig * np.sqrt(tf)  # Compute total log-return standard deviation sigma * sqrt(tau_fix)
    d1 = lambda K: (np.log(fwd / K) + 0.5 * vol_sqrt_t ** 2) / vol_sqrt_t  # Define Black's formula d1(K) function
    d2 = lambda K: (np.log(fwd / K) - 0.5 * vol_sqrt_t ** 2) / vol_sqrt_t  # Define Black's formula d2(K) function
    bond_pv = DENOM * df_credit  # Present value of the zero-coupon bond paying 1,000 USD at redemption
    digital_pv = BONUS_COUPON * df_credit * norm.cdf(d2(INIT_FIX))  # Present value of USD 86 binary call struck at barrier H
    call_pv = CALL_UNITS * df_credit * (fwd * norm.cdf(d1(STRIKE_BONUS)) - STRIKE_BONUS * norm.cdf(d2(STRIKE_BONUS)))  # Present value of 22.163 calls
    return bond_pv, digital_pv, call_pv  # Return tuple of component present values (bond, digital, outperformance call)


def exercise_threshold(r, q, s, sig):  # Solve for critical TLT price S* at T23 where issuer is indifferent to calling
    call_redemp_pv = EARLY_REDEMP_AMT * np.exp(-(r + s) * year_frac(T23, PAY23))  # Discount 1,086 USD from PAY23 back to T23
    obj = lambda x: float(sum(non_callable_parts(x, r, q, s, sig, T23))) - call_redemp_pv  # Define root equation C23(x) - R*delta_23 = 0
    return brentq(obj, 1.0, 500.0)  # Solve for unique root S* on [1.0, 500.0] USD via Brent's method


def value_certificate(S, r, q, s, sig, t0, n_grid=2001):  # Value full callable certificate via European reduction at T23
    b, d, c = non_callable_parts(S, r, q, s, sig, t0)  # Compute present values of the 3 non-callable components at t0
    nc_val = float(b + d + c)  # Sum components to obtain total non-callable certificate value V_NC
    tau23 = year_frac(t0, T23)  # Compute year fraction from valuation date t0 to final call observation date T23
    z = np.linspace(-7.0, 7.0, n_grid)  # Build uniform standard normal quadrature grid over [-7, 7]
    weights = norm.pdf(z)  # Evaluate standard normal density phi(z) at every grid node
    weights /= weights.sum()  # Normalize density weights so they integrate to 1.0
    S_T23 = S * np.exp((r - q - 0.5 * sig ** 2) * tau23 + sig * np.sqrt(tau23) * z)  # Map z-grid to lognormal TLT prices at T23
    b2, d2, c2 = non_callable_parts(S_T23, r, q, s, sig, T23)  # Price all 3 continuation components at T23 across the grid
    cont_T23 = b2 + d2 + c2  # Compute continuation value C23(S_T23) at every grid node
    call_redemp_pv = EARLY_REDEMP_AMT * np.exp(-(r + s) * year_frac(T23, PAY23))  # Discounted early redemption amount at T23
    berm_val = float(np.exp(-(r + s) * tau23) * np.sum(weights * np.maximum(cont_T23 - call_redemp_pv, 0.0)))  # Integrate issuer call value
    s_star = exercise_threshold(r, q, s, sig)  # Compute exact critical TLT price S* at T23 above which issuer calls
    fwd_23 = S * np.exp((r - q) * tau23)  # Compute risk-neutral forward price of TLT to T23
    d_star = (np.log(fwd_23 / s_star) - 0.5 * sig ** 2 * tau23) / (sig * np.sqrt(tau23))  # Compute d2(S*) for Q(S_T23 > S*)
    call_prob = float(norm.cdf(d_star))  # Compute analytical risk-neutral probability of early call at T23
    return dict(bond=float(b), dig=float(d), call=float(c), nc=nc_val, berm=berm_val,  # Return valuation dictionary with components,
                cert=nc_val - berm_val, pcall=call_prob)  # non-callable total, issuer call, net certificate value, and call probability


def binomial_tree_check(S, r, q, s, sig, t0, steps_per_day=2):  # Independently verify Bermudan valuation on a CRR binomial tree
    total_days = (FINAL_FIX_DATE - t0).days  # Compute total calendar days from valuation date t0 to final fixing date
    n_steps = total_days * steps_per_day  # Set total number of tree steps (2 steps per calendar day)
    dt = 1.0 / (365.0 * steps_per_day)  # Compute step duration dt in years
    up = np.exp(sig * np.sqrt(dt))  # Compute CRR up factor u = exp(sigma * sqrt(dt))
    down = 1.0 / up  # Compute CRR down factor d = 1 / u so the tree recombines
    p_up = (np.exp((r - q) * dt) - down) / (up - down)  # Compute risk-neutral up-move probability adjusted for yield q
    disc_step = np.exp(-(r + s) * dt)  # Compute one-step credit-adjusted discount factor
    idx = np.arange(n_steps + 1)  # Create index array of downward move counts (0 to n_steps) at maturity
    S_T = S * up ** (n_steps - 2 * idx)  # Compute TLT spot prices across all terminal nodes at FINAL_FIX_DATE
    V = np.exp(-(r + s) * year_frac(FINAL_FIX_DATE, REDEMP_DATE)) * (  # Discount terminal certificate payoff from REDEMP_DATE to FINAL_FIX_DATE:
        DENOM + BONUS_COUPON * (S_T > INIT_FIX) + CALL_UNITS * np.maximum(S_T - STRIKE_BONUS, 0.0))  # bond + binary bonus + outperformance call
    obs_map = {(o - t0).days * steps_per_day: (k + 1, EARLY_REDEMP_AMT * np.exp(-(r + s) * year_frac(o, pay)))  # Map step index to
               for k, (o, pay) in enumerate(OBS_SCHEDULE) if o > t0}  # (observation number k, discounted call payout R * delta_k)
    exercise_counts = {}  # Initialize dictionary tracking how many nodes trigger issuer exercise on each observation date
    ex_at_last = None  # Placeholder for binary exercise indicator array at the final observation date T23
    last_step = max(obs_map)  # Find step index corresponding to the final (23rd) observation date
    for j in range(n_steps - 1, -1, -1):  # Perform backward induction from step n_steps - 1 down to root j = 0
        V = disc_step * (p_up * V[:j + 1] + (1.0 - p_up) * V[1:j + 2])  # Compute discounted risk-neutral continuation value at step j
        if j in obs_map:  # Check whether current time step j is an issuer call observation date
            k, call_pv = obs_map[j]  # Unpack observation index k and discounted call payout
            exercised = call_pv < V  # Identify nodes where calling is cheaper for the issuer than continuing
            exercise_counts[k] = int(exercised.sum())  # Record number of exercised nodes for observation date k
            if j == last_step:  # If at the final observation date T23 (k = 23)
                ex_at_last = exercised.astype(float)  # Store 0/1 exercise indicators to propagate Q-probability back to t0
            V = np.minimum(V, call_pv)  # Apply issuer Bellman optimality condition V_k = min(continuation, R * delta_k)
    prob_vec = ex_at_last  # Initialize probability array at last_step with exercise indicators
    for j in range(last_step - 1, -1, -1):  # Propagate risk-neutral exercise probability backward to step j = 0
        prob_vec = p_up * prob_vec[:j + 1] + (1.0 - p_up) * prob_vec[1:j + 2]  # Compute conditional expectation of exercise indicator
    return float(V[0]), exercise_counts, float(prob_vec[0])  # Return root tree price, exercise node counts, and tree call probability


# =====================================================================
# 4) VOLATILITY ESTIMATION: HISTORICAL, EWMA, GARCH(1,1)
# =====================================================================
def vol_historical(u, window=252):  # Compute equally weighted annualized historical volatility over last 252 days
    return float(np.std(u[-window:], ddof=1) * np.sqrt(252))  # Sample standard deviation scaled by sqrt(252)


def vol_ewma(u, lam=EWMA_LAMBDA):  # Compute Exponentially Weighted Moving Average annualized volatility
    var_t = float(np.var(u[:30]))  # Initialize daily variance recursion using the first 30 observations
    for ret_i in u:  # Iterate chronologically over each daily log-return u_i
        var_t = lam * var_t + (1.0 - lam) * ret_i * ret_i  # Update EWMA variance: sigma_n^2 = lambda * sigma_{n-1}^2 + (1 - lambda) * u_{n-1}^2
    return float(np.sqrt(252 * var_t))  # Return annualized EWMA volatility at the end of the sample


def fit_garch_mle(u):  # Calibrate GARCH(1,1) via unconstrained reparameterized Gaussian MLE
    """Unconstrained reparameterized GARCH(1,1) MLE (strictly positive stationary domain)."""  # Function docstring
    u_pct = np.asarray(u, float) * 100.0  # Scale daily log-returns to percentages for numerical conditioning
    var_init = float(np.var(u_pct))  # Compute sample variance in percent squared to initialize conditional variance

    def unpack(z):  # Map unconstrained vector z in R^3 strictly inside the stationary positive GARCH domain
        z1, z2, z3 = np.clip(z, -20.0, 20.0)  # Clip z to [-20, 20] to avoid floating-point overflow in exp/expit
        pers = 0.9995 * float(expit(z1))  # Map z1 into persistence (alpha + beta) in (0, 0.9995)
        share = float(expit(z2))  # Map z2 into alpha's share of persistence in (0, 1)
        alpha = pers * share  # Recover strictly positive ARCH weight alpha
        beta = pers * (1.0 - share)  # Recover strictly positive GARCH weight beta with alpha + beta < 0.9995
        vl_pct = float(np.exp(z3))  # Map z3 into strictly positive long-run variance V_L (in %^2)
        return alpha, beta, vl_pct  # Return valid stationary parameters (alpha, beta, V_L)

    def neg_log_lik(z):  # Compute negative Gaussian log-likelihood (excluding additive constants) at z
        alpha, beta, vl_pct = unpack(z)  # Transform unconstrained optimizer coordinates to GARCH parameters
        omega_pct = vl_pct * (1.0 - alpha - beta)  # Compute strictly positive intercept omega = (1 - alpha - beta) * V_L
        s2 = var_init  # Initialize conditional variance recursion at unconditional sample variance
        nll_sum = 0.0  # Initialize negative log-likelihood accumulator
        for x_i in u_pct:  # Loop over all percentage log-returns in the estimation window
            nll_sum += np.log(s2) + (x_i * x_i) / s2  # Add day-i contribution ln(sigma_i^2) + u_i^2 / sigma_i^2
            s2 = omega_pct + alpha * (x_i * x_i) + beta * s2  # Update conditional variance for the next trading day
        return nll_sum  # Return total negative log-likelihood value

    best_res = None  # Track best optimization result across multiple starting guesses
    log_v0 = float(np.log(max(var_init, 1e-6)))  # Initial unconstrained coordinate z3 = ln(sample variance)
    for a0, b0 in [(0.05, 0.90), (0.10, 0.85), (0.03, 0.95)]:  # Loop over three diverse starting pairs (alpha_0, beta_0)
        p0 = (a0 + b0) / 0.9995  # Convert initial persistence to (0, 1) fraction for logit transform
        sh0 = a0 / (a0 + b0)  # Convert initial alpha share to (0, 1) fraction for logit transform
        z0 = [np.log(p0 / (1.0 - p0)), np.log(sh0 / (1.0 - sh0)), log_v0]  # Build unconstrained starting vector z0
        res = minimize(neg_log_lik, z0, method="BFGS")  # Run unconstrained BFGS optimization without boundary clipping
        if best_res is None or res.fun < best_res.fun:  # Retain the candidate achieving the lowest negative log-likelihood
            best_res = res  # Update best optimization result

    alpha_hat, beta_hat, vl_pct_hat = unpack(best_res.x)  # Unpack optimal MLE parameters from best unconstrained solution
    vl_daily = vl_pct_hat / 1e4  # Convert long-run daily variance V_L from %^2 back to decimal units
    omega_daily = vl_daily * (1.0 - alpha_hat - beta_hat)  # Compute decimal intercept omega = gamma * V_L
    s2_curr = var_init / 1e4  # Initialize decimal conditional variance at start of sample
    for ret_i in u:  # Filter historical daily returns forward to compute one-step-ahead variance at valuation date
        s2_curr = omega_daily + alpha_hat * (ret_i * ret_i) + beta_hat * s2_curr  # Update conditional variance sigma_n^2
    return dict(omega=omega_daily, alpha=alpha_hat, beta=beta_hat, vl=vl_daily, s2n=s2_curr, n=len(u))  # Return calibrated dict


def garch_term_structure(alpha, beta, vl, s2n, n_days):  # Compute annualized GARCH(1,1) average volatility over option life
    persistence = alpha + beta  # Compute variance persistence (alpha + beta)
    if 1.0 - persistence < 1e-10:  # Guard against division by zero in the unit-root limit (alpha + beta -> 1)
        avg_daily_var = s2n  # Without mean reversion, expected average variance equals current variance s2n
    else:  # Standard stationary GARCH(1,1) closed-form geometric average
        avg_daily_var = vl + (1.0 - persistence ** n_days) / (n_days * (1.0 - persistence)) * (s2n - vl)  # (1/N)*sum E[sigma_{n+k}^2]
    return float(np.sqrt(252 * avg_daily_var))  # Annualize expected average daily variance using 252 trading days


# =====================================================================
# 5) DATA LOADING
# =====================================================================
def load_tlt_data(csv_path):  # Load TLT daily prices from a CSV file or yfinance and compute log-returns
    import pandas as pd  # Import pandas locally for tabular time-series handling
    try:  # Wrap external I/O and yfinance download in try/except for graceful fallback
        if csv_path:  # If user provided a local CSV file path via --data
            df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")  # Read CSV with Date parsed as DatetimeIndex
        else:  # Otherwise download TLT daily history directly from Yahoo Finance
            import yfinance as yf_client  # Import yfinance library
            df = yf_client.download("TLT", start="2015-01-01", auto_adjust=False, progress=False)  # Download OHLCV history
            if hasattr(df.columns, "droplevel") and df.columns.nlevels > 1:  # Check if yfinance returned MultiIndex columns
                df.columns = df.columns.droplevel(1)  # Drop ticker level from MultiIndex to get flat column names
        adj_col = "Adj Close" if "Adj Close" in df.columns else "Close"  # Prefer dividend-adjusted close; fall back to raw close
        adj_series = df[adj_col].dropna().sort_index()  # Drop missing rows and sort in ascending date order
        log_rets = np.log(adj_series / adj_series.shift(1)).dropna()  # Compute daily log-returns u_i = ln(S_i / S_{i-1})
        close_series = df["Close"].dropna().sort_index()  # Extract raw closing prices for spot lookups
        return log_rets, close_series  # Return tuple of (log_returns, close_prices)
    except Exception as exc:  # Catch missing package, network error, or missing file
        print(f"[!] TLT data not loaded ({exc}). Using fallback GARCH parameters.")  # Print warning to console
        return None, None  # Return (None, None) to activate fallback parameters


# =====================================================================
# 6) LATEX FORMATTING HELPERS
# =====================================================================
def fmt_num(x, nd=2, math_mode=False):  # Format float x to nd decimals for LaTeX text or math mode
    if x < 0:  # Handle negative numbers with proper LaTeX minus sign
        return ("-" if math_mode else "$-$") + fmt_num(-x, nd, math_mode)  # Prepend minus and format absolute value
    return f"{x:,.{nd}f}".replace(",", r"\,")  # Use thin space \, as thousands separator and '.' as decimal point


def fmt_pct(x, nd=2, math_mode=False):  # Format decimal x as a LaTeX percentage string
    return fmt_num(100.0 * x, nd, math_mode) + r"\,\%"  # Multiply by 100 and append thin space + escaped percent sign


def fmt_date(d):  # Format datetime.date object as 'D Mon YYYY' (e.g., '3 Dec 2025')
    return d.strftime("%d %b %Y").lstrip("0")  # Strip leading zero from day of month


def fmt_sci(x):  # Format very small floats in LaTeX scientific notation a \cdot 10^{b}
    mantissa, exponent = f"{x:.3e}".split("e")  # Split Python scientific string into mantissa and exponent
    return f"{mantissa}\\cdot 10^{{{int(exponent)}}}"  # Return LaTeX math-mode scientific notation string


# =====================================================================
# 7) MAIN EXECUTION AND LATEX EXPORT
# =====================================================================
def main():  # Main workflow coordinating CLI parsing, volatility estimation, valuation, and LaTeX output
    parser = argparse.ArgumentParser(description="1BP451 Financial Derivatives II Certificate Valuation")  # Create CLI parser
    parser.add_argument("--data", help="Path to Yahoo Finance CSV file (Date, Close, Adj Close)")  # Optional --data flag
    parser.add_argument("--out", default="outputs.tex", help="Output LaTeX macro file path")  # Optional --out flag
    args = parser.parse_args()  # Parse provided command-line arguments
    log_rets, close_prices = load_tlt_data(args.data)  # Load TLT historical returns and closing prices
    is_estimated = log_rets is not None  # Set boolean flag indicating empirical MLE estimation succeeded

    results = {}  # Initialize dictionary storing valuation results per scenario key ("A", "B", "C")
    for scen in VALUATIONS:  # Iterate over all configured valuation dates
        k, t0, S, r, q, s = scen["key"], scen["val_date"], scen["S"], scen["r"], scen["q"], scen["s"]  # Unpack scenario inputs
        if S is None and close_prices is not None:  # Auto-fill spot price S from close series if omitted
            S = float(close_prices[close_prices.index <= np.datetime64(t0)].iloc[-1])  # Lookup latest close <= t0
        if S is None or r is None or q is None:  # Skip scenario if any required numeric input is still missing
            continue  # Proceed to next scenario

        n_bus_days = int(round(252 * year_frac(t0, FINAL_FIX_DATE)))  # Compute remaining trading days N to final fixing
        info = dict(scen, S=S, N=n_bus_days)  # Create result dictionary for scenario k initialized with inputs and N
        if is_estimated:  # If empirical return series was loaded
            u_win = log_rets[log_rets.index <= np.datetime64(t0)].iloc[-EST_WINDOW:].values  # Take last 1,260 returns up to t0
            g_params = fit_garch_mle(u_win)  # Estimate GARCH(1,1) parameters via unconstrained BFGS MLE
            info["hist"] = vol_historical(u_win)  # Compute 252-day historical volatility
            info["ewma"] = vol_ewma(u_win)  # Compute EWMA volatility (lambda = 0.97)
        else:  # Otherwise populate fallback GARCH(1,1) parameters
            vl_d = FALLBACK_GARCH["vl_ann"] ** 2 / 252.0  # Convert fallback long-run annual vol to daily variance
            a_fb, b_fb = FALLBACK_GARCH["alpha"], FALLBACK_GARCH["beta"]  # Unpack fallback alpha and beta
            g_params = dict(omega=vl_d * (1.0 - a_fb - b_fb), alpha=a_fb, beta=b_fb, vl=vl_d,  # Assemble fallback dict
                            s2n=FALLBACK_SIGN[k] ** 2 / 252.0, n=EST_WINDOW)  # Set fallback spot daily variance
            info["hist"] = None  # No rolling historical vol available in fallback mode
            info["ewma"] = None  # No EWMA vol available in fallback mode

        info["g"] = g_params  # Save calibrated GARCH(1,1) parameter dictionary
        info["sig_n"] = float(np.sqrt(252 * g_params["s2n"]))  # Annualize filtered spot volatility sigma_n
        info["vl_ann"] = float(np.sqrt(252 * g_params["vl"]))  # Annualize long-run volatility sqrt(252 * V_L)
        info["sig_ts"] = garch_term_structure(g_params["alpha"], g_params["beta"], g_params["vl"], g_params["s2n"], n_bus_days)  # GARCH-TS vol

        vol_models = [  # Assemble list of (column_label, volatility) pairs for comparison tables
            ("Hist.\\ 252d", info["hist"]),  # 252-day historical volatility
            ("EWMA", info["ewma"]),  # EWMA volatility
            ("GARCH $\\sigma_n$", info["sig_n"]),  # Spot GARCH volatility
            ("GARCH-TS", info["sig_ts"]),  # GARCH volatility term structure (main model)
        ]  # End of volatility model list
        info["models"] = [(label, sig_val) for label, sig_val in vol_models if sig_val is not None]  # Keep available models
        info["res"] = [value_certificate(S, r, q, s, sig_val, t0) for _, sig_val in info["models"]]  # Price certificate per vol model
        base_res = value_certificate(S, r, q, s, info["sig_ts"], t0)  # Compute primary valuation using GARCH-TS volatility
        info["base"] = base_res  # Store primary valuation result dictionary

        info["sens_table"] = [  # Compute certificate value under alternative modeling assumptions
            ("Base model (GARCH-TS)", base_res["cert"]),  # Full baseline model
            ("No dividends ($q=0$)", value_certificate(S, r, 0.0, s, info["sig_ts"], t0)["cert"]),  # Ignore ETF dividends
            ("No credit spread ($s=0$)", value_certificate(S, r, q, 0.0, info["sig_ts"], t0)["cert"]),  # Ignore BIL credit risk
            ("Volatility $+2$ pp", value_certificate(S, r, q, s, info["sig_ts"] + 0.02, t0)["cert"]),  # +2 percentage point vol bump
            ("Volatility $-2$ pp", value_certificate(S, r, q, s, info["sig_ts"] - 0.02, t0)["cert"]),  # -2 percentage point vol bump
            ("Non-callable ($V^{\\rm NC}$)", base_res["nc"]),  # Ignore issuer Bermudan call option
        ]  # End of modeling sensitivity table list

        dS = 0.5  # Set spot bump size (0.50 USD) for central finite-difference Delta
        delta = (value_certificate(S + dS, r, q, s, info["sig_ts"], t0)["cert"]  # Value at S + dS
                 - value_certificate(S - dS, r, q, s, info["sig_ts"], t0)["cert"]) / (2.0 * dS)  # Minus value at S - dS over 2*dS
        vega = value_certificate(S, r, q, s, info["sig_ts"] + 0.01, t0)["cert"] - base_res["cert"]  # Vega per +1 pp volatility increase
        bpv = value_certificate(S, r - 0.0001, q, s, info["sig_ts"], t0)["cert"] - base_res["cert"]  # Basis Point Value V(r - 1bp) - V(r)
        info["greeks"] = (delta, vega, bpv)  # Store risk sensitivity tuple (Delta, Vega, BPV)

        tree_val, tree_ex, tree_prob = binomial_tree_check(S, r, q, s, info["sig_ts"], t0)  # Run independent CRR binomial tree check
        info["tree"] = (tree_val, tree_ex, tree_prob)  # Store binomial tree check outputs
        info["s_star"] = exercise_threshold(r, q, s, info["sig_ts"])  # Store critical early-call threshold S* at T23

        info["impl_vol"] = None  # Initialize Leonteq implied volatility to None
        quote = LEONTEQ_QUOTES.get(k)  # Look up indicative Leonteq quote for scenario k
        if quote is not None:  # If a market quote is provided
            mid_usd = 0.5 * (quote[0] + quote[1]) * DENOM / 100.0  # Convert mid quote (%) to USD per 1,000 notional
            iv_obj = lambda sg: value_certificate(S, r, q, s, sg, t0)["cert"] - mid_usd  # Define implied-vol root objective
            try:  # Attempt to invert certificate price for implied volatility on [2%, 60%]
                info["impl_vol"] = brentq(iv_obj, 0.02, 0.60)  # Solve for implied volatility via Brent's method
            except ValueError:  # Catch case where market price lies outside model volatility range
                info["impl_vol"] = None  # Leave implied volatility undefined ("--")

        results[k] = info  # Store completed scenario results dictionary under key k
        ex_dates = {obs_k: nodes for obs_k, nodes in tree_ex.items() if nodes > 0}  # Extract observation dates with active exercise
        print(f"{k} {t0}: sigma_n={info['sig_n']:.4f} TS={info['sig_ts']:.4f} "  # Print diagnostic summary line to console:
              f"cert={base_res['cert']:.2f} ({base_res['cert']/10:.2f}%) NC={base_res['nc']:.2f} "  # model value and non-callable value,
              f"berm={base_res['berm']:.2f} P_call={base_res['pcall']:.3f} | tree={tree_val:.2f} "  # issuer call, call prob, tree value,
              f"exercise(k:nodes)={ex_dates} P_tree={tree_prob:.3f}")  # exercised dates and tree call probability

    write_latex(args.out, results, is_estimated)  # Export all macros and tables to outputs.tex
    print(f"Done -> {args.out}")  # Confirm completion in console


def write_latex(out_path, results, is_estimated):  # Generate outputs.tex containing all LaTeX macros and tables for main.tex
    lines = []  # Initialize list of output LaTeX lines
    add = lines.append  # Bind local shorthand for appending lines
    keys = list(results)  # Extract ordered list of evaluated scenario keys (["A", "B", "C"])

    def macro(name, body):  # Helper appending a single-line LaTeX \newcommand definition
        add(f"\\newcommand{{\\{name}}}{{{body}}}")  # Format and append \newcommand{\name}{body}

    if is_estimated:  # Define \GarchStatus sentence for empirical MLE run
        macro("GarchStatus", f"Parameters are estimated via MLE on the last {EST_WINDOW} daily dividend-adjusted log-returns of TLT prior to each valuation date.")
    else:  # Define \GarchStatus sentence for fallback run
        macro("GarchStatus", f"GARCH(1,1) parameters over a {EST_WINDOW}-day daily return window.")

    for k in keys:  # Loop over scenarios A, B, C to define scalar macros referenced in prose
        i = results[k]  # Retrieve scenario data dictionary
        b = i["base"]  # Retrieve baseline GARCH-TS valuation outputs
        direction = "downward-sloping" if i["sig_n"] > i["vl_ann"] else "upward-sloping"  # Determine slope of volatility term structure
        macro(f"Dir{k}", direction)  # Macro \DirA / \DirB / \DirC: term-structure slope label
        macro(f"c{k}", fmt_pct(b["cert"] / DENOM, 2))  # Macro \cA / \cB / \cC: net certificate value in % of notional
        macro(f"nc{k}", fmt_num(b["nc"]))  # Macro \ncA / \ncB / \ncC: non-callable value in USD
        macro(f"ncp{k}", fmt_pct(b["nc"] / DENOM, 2))  # Macro \ncpA / \ncpB / \ncpC: non-callable value in % of notional
        macro(f"berm{k}", fmt_num(b["berm"]))  # Macro \bermA / \bermB / \bermC: issuer Bermudan call value in USD
        macro(f"prob{k}", fmt_pct(b["pcall"], 1))  # Macro \probA / \probB / \probC: risk-neutral call probability in %
        macro(f"gap{k}", fmt_num(100.0 - b["cert"] / 10.0, 2))  # Macro \gapA / \gapB / \gapC: difference vs. 100% par
        macro(f"sigts{k}", fmt_pct(i["sig_ts"], 2))  # Macro \sigtsA / \sigtsB / \sigtsC: GARCH-TS annualized volatility
        macro(f"sign{k}", fmt_pct(i["sig_n"], 2))  # Macro \signA / \signB / \signC: spot GARCH annualized volatility
        macro(f"sigvl{k}", fmt_pct(i["vl_ann"], 2))  # Macro \sigvlA / \sigvlB / \sigvlC: long-run GARCH volatility
        dist_pct = (i["S"] / INIT_FIX - 1.0) * 100.0  # Compute percentage distance of spot TLT price from barrier H
        macro(f"Pos{k}", f"{fmt_num(abs(dist_pct), 2)}\\,\\% {'below' if dist_pct < 0 else 'above'} the barrier")  # Macro \PosA..\PosC
        macro(f"Sstar{k}", fmt_num(i["s_star"]))  # Macro \SstarA / \SstarB / \SstarC: critical threshold S* at T23
        t_val, t_ex, t_prob = i["tree"]  # Unpack CRR binomial tree verification results
        macro(f"tree{k}", fmt_pct(t_val / DENOM, 2))  # Macro \treeA / \treeB / \treeC: tree price in % of notional
        macro(f"treep{k}", fmt_pct(t_prob, 1))  # Macro \treepA / \treepB / \treepC: tree call probability in %
        macro(f"date{k}", fmt_date(i["val_date"]))  # Macro \dateA / \dateB / \dateC: formatted valuation date
        macro(f"exobs{k}", ", ".join(str(obs_k) for obs_k, cnt in t_ex.items() if cnt > 0) or "none")  # Macro \exobsA..\exobsC

    col_spec = "c" * len(keys)  # Build centered column specification string ("ccc" for 3 scenarios)
    header_dates = " & ".join(fmt_date(results[k]["val_date"]) for k in keys)  # Build LaTeX column header row of dates

    def row(label, fn):  # Helper formatting a single table row across all scenario columns
        return f"{label} & " + " & ".join(fn(results[k]) for k in keys) + r" \\"  # Join formatted entries with & and trailing \\

    opt_pct = lambda x: "--" if x is None else fmt_pct(x, 2)  # Format optional percentage value or double dash

    add(r"\newcommand{\TabInputs}{\begin{table}[ht]\centering\small")  # Start \TabInputs table macro
    add(r"\caption{Market inputs and volatility estimates.}\label{tab:inputs}")  # Caption and cross-reference label
    add(f"\\begin{{tabular}}{{@{{}}l{col_spec}@{{}}}}\\toprule")  # Open tabular environment with top rule
    add(f" & {header_dates} \\\\\\midrule")  # Write valuation date headers and midrule
    add(row("TLT spot $S_t$ (USD)", lambda i: fmt_num(i["S"])))  # Row: TLT spot price
    add(row(r"$\tau_{\rm fix}$ (yr) / $N$ days",  # Row: time to fixing in years and trading days
            lambda i: f"{fmt_num(year_frac(i['val_date'], FINAL_FIX_DATE), 3)} / {i['N']}"))  # Format tau_fix / N
    add(row(r"$\tau_{\rm pay}$ (yr)", lambda i: fmt_num(year_frac(i["val_date"], REDEMP_DATE), 3)))  # Row: time to redemption
    add(row("Risk-free rate $r$", lambda i: fmt_pct(i["r"], 2)))  # Row: risk-free rate r
    add(row("Dividend yield $q$", lambda i: fmt_pct(i["q"], 2)))  # Row: dividend yield q
    add(row("BIL credit spread $s$", lambda i: fmt_pct(i["s"], 2)))  # Row: credit spread s
    add(r"\midrule")  # Horizontal rule separating market inputs from volatility estimates
    add(row("Hist.\\ vol (252d)", lambda i: opt_pct(i["hist"])))  # Row: 252-day historical volatility
    add(row(f"EWMA ($\\lambda={EWMA_LAMBDA}$)", lambda i: opt_pct(i["ewma"])))  # Row: EWMA volatility
    add(row(r"GARCH spot $\sigma_n$", lambda i: fmt_pct(i["sig_n"], 2)))  # Row: spot GARCH volatility
    add(row(r"GARCH long-run $\sqrt{252V_L}$", lambda i: fmt_pct(i["vl_ann"], 2)))  # Row: long-run GARCH volatility
    add(row(r"GARCH-TS $\sigma(N)$", lambda i: fmt_pct(i["sig_ts"], 2)))  # Row: GARCH term-structure volatility
    add(r"\bottomrule\end{tabular}\end{table}}")  # Close \TabInputs table macro

    add(r"\newcommand{\TabGarch}{\begin{table}[ht]\centering\small")  # Start \TabGarch table macro
    add(r"\caption{GARCH(1,1) MLE parameters (daily returns).}\label{tab:garch}")  # Caption and label
    add(f"\\begin{{tabular}}{{@{{}}l{col_spec}@{{}}}}\\toprule")  # Open tabular environment
    add(f" & {header_dates} \\\\\\midrule")  # Date column headers
    add(row(r"$\omega$", lambda i: "$" + fmt_sci(i["g"]["omega"]) + "$"))  # Row: omega intercept
    add(row(r"$\alpha$", lambda i: fmt_num(i["g"]["alpha"], 4)))  # Row: ARCH parameter alpha
    add(row(r"$\beta$", lambda i: fmt_num(i["g"]["beta"], 4)))  # Row: GARCH parameter beta
    add(row(r"$\alpha+\beta$", lambda i: fmt_num(i["g"]["alpha"] + i["g"]["beta"], 4)))  # Row: persistence alpha + beta
    add(row(r"$\gamma=1-\alpha-\beta$", lambda i: fmt_num(1.0 - i["g"]["alpha"] - i["g"]["beta"], 4)))  # Row: mean-reversion speed
    add(row(r"$V_L$ (daily var.)", lambda i: "$" + fmt_sci(i["g"]["vl"]) + "$"))  # Row: long-run daily variance V_L
    add(row("Sample size", lambda i: "--" if i["g"]["n"] is None else str(i["g"]["n"])))  # Row: estimation window length
    add(r"\bottomrule\end{tabular}\end{table}}")  # Close \TabGarch table macro

    for k in keys:  # Loop over scenarios A, B, C to build per-date valuation tables (\TabResA, \TabResB, \TabResC)
        i = results[k]  # Retrieve scenario data
        m_list = i["models"]  # Retrieve list of volatility models evaluated for scenario k
        add(f"\\newcommand{{\\TabRes{k}}}{{\\begin{{table}}[ht]\\centering\\small")  # Start \TabResA / \TabResB / \TabResC
        add(f"\\caption{{Valuation as of {fmt_date(i['val_date'])} ($S_t={fmt_num(i['S'], 2, True)}$ USD; per 1,000 USD).}}"
            f"\\label{{tab:res{k}}}")  # Caption and label for scenario k
        add(f"\\begin{{tabular}}{{@{{}}l{'c' * len(m_list)}@{{}}}}\\toprule")  # Open tabular sized to number of vol models
        add(" & " + " & ".join(lbl for lbl, _ in m_list) + r" \\")  # Model name column headers
        add(r"\midrule")  # Midrule below model headers
        add(r"$\sigma$ (p.a.) & " + " & ".join(fmt_pct(sg, 2) for _, sg in m_list) + r" \\\midrule")  # Row: annualized volatilities

        def comp_row(lbl, dict_key, nd_=2, negate=False):  # Helper formatting one valuation component row across models
            return lbl + " & " + " & ".join(fmt_num(-x[dict_key] if negate else x[dict_key], nd_) for x in i["res"]) + r" \\"

        add(comp_row("Zero-coupon bond", "bond"))  # Row: zero-coupon bond PV
        add(comp_row("Digital call (USD 86)", "dig"))  # Row: digital bonus call PV
        add(comp_row(f"Outperformance call ({fmt_num(CALL_UNITS, 3)}$\\times$)", "call"))  # Row: outperformance call PV
        add(r"\midrule")  # Midrule before non-callable subtotal
        add(comp_row(r"$V^{\rm NC}$ (non-callable)", "nc"))  # Row: non-callable total in USD
        add(r"\quad \% notional & " + " & ".join(fmt_num(x["nc"] / 10.0, 2) for x in i["res"]) + r" \\")  # Row: non-callable in %
        add(comp_row(r"Issuer call ($-V^{\rm Berm}$)", "berm", negate=True))  # Row: issuer Bermudan call deduction
        add(r"$Q$-prob.\ call (25 Oct 2027) & "  # Row: risk-neutral probability of early call on 25 Oct 2027
            + " & ".join(fmt_pct(x["pcall"], 1) for x in i["res"]) + r" \\\midrule")  # Format call probabilities across models
        add(comp_row(r"$V^{\rm Cert}$ (certificate)", "cert"))  # Row: net callable certificate value in USD
        add(r"\quad \% notional & " + " & ".join(r"\textbf{" + fmt_num(x["cert"] / 10.0, 2) + "}" for x in i["res"]) + r" \\")  # Bold %
        add(r"\bottomrule\end{tabular}\end{table}}")  # Close \TabRes{k} table macro

    add(r"\newcommand{\TabSens}{\begin{table}[ht]\centering\small")  # Start \TabSens modeling sensitivity table macro
    add(r"\caption{Modeling choices sensitivity (GARCH-TS, \% notional).}\label{tab:sens}")  # Caption and label
    add(f"\\begin{{tabular}}{{@{{}}l{col_spec}@{{}}}}\\toprule")  # Open tabular environment
    add(f" & {header_dates} \\\\\\midrule")  # Date column headers
    for j in range(len(results[keys[0]]["sens_table"])):  # Loop over all 6 modeling sensitivity rows
        lbl = results[keys[0]]["sens_table"][j][0]  # Extract row label from first scenario
        add(f"{lbl} & " + " & ".join(fmt_num(results[k]["sens_table"][j][1] / 10.0, 2) for k in keys) + r" \\")  # Values in %
    add(r"\bottomrule\end{tabular}\end{table}}")  # Close \TabSens table macro

    add(r"\newcommand{\TabGreeks}{\begin{table}[ht]\centering\small")  # Start \TabGreeks risk sensitivity table macro
    add(r"\caption{Risk sensitivities (GARCH-TS).}\label{tab:greeks}")  # Caption and label
    add(f"\\begin{{tabular}}{{@{{}}l{col_spec}@{{}}}}\\toprule")  # Open tabular environment
    add(f" & {header_dates} \\\\\\midrule")  # Date column headers
    add(row("Delta (USD / $+1$ USD $S$)", lambda i: fmt_num(i["greeks"][0], 3)))  # Row: Delta
    add(row(r"Vega (USD / $+1$ pp $\sigma$)", lambda i: fmt_num(i["greeks"][1], 2)))  # Row: Vega
    add(row(r"BPV $=V(r-0.01\%)-V(r)$", lambda i: fmt_num(i["greeks"][2], 3)))  # Row: Basis Point Value
    add(r"\bottomrule\end{tabular}\end{table}}")  # Close \TabGreeks table macro

    quoted_keys = [k for k in keys if LEONTEQ_QUOTES.get(k) is not None]  # Collect evaluated scenarios that have Leonteq quotes
    if quoted_keys:  # Build \LeonteqTable comparison table if quotes are present
        add(r"\newcommand{\LeonteqTable}{\begin{table}[ht]\centering\small")  # Start \LeonteqTable macro
        add(r"\caption{Model vs.\ Leonteq quotes (\% notional).}\label{tab:leonteq}")  # Caption and label
        add(r"\begin{tabular}{@{}lcccc@{}}\toprule")  # Open 5-column tabular environment
        add(r"Date & Leonteq & Model (GARCH-TS) & Diff.\ (pp) & Implied $\sigma$ \\\midrule")  # Column headers
        for k in quoted_keys:  # Iterate over quoted scenarios
            i, qt = results[k], LEONTEQ_QUOTES[k]  # Retrieve scenario outputs and quote pair
            q_mid = 0.5 * (qt[0] + qt[1])  # Compute mid quote in % of notional
            m_pct = i["base"]["cert"] / 10.0  # Convert model price to % of notional
            add(f"{fmt_date(i['val_date'])} & {fmt_num(q_mid)} & {fmt_num(m_pct)} & {fmt_num(m_pct - q_mid, 2, True)} & "
                f"{'--' if i['impl_vol'] is None else fmt_pct(i['impl_vol'], 2)} \\\\")  # Write formatted comparison row
        add(r"\bottomrule\end{tabular}\end{table}}")  # Close \LeonteqTable macro
    else:  # Define empty macro if no Leonteq quotes exist
        macro("LeonteqTable", "")  # Empty fallback macro

    k_ex = keys[1] if len(keys) > 1 else keys[0]  # Select Scenario B (23 Oct 2026) for Appendix A worked example
    i = results[k_ex]  # Retrieve Scenario B outputs
    S, r, q, s, t0, sig = i["S"], i["r"], i["q"], i["s"], i["val_date"], i["sig_ts"]  # Unpack inputs for worked example
    tf, tp = year_frac(t0, FINAL_FIX_DATE), year_frac(t0, REDEMP_DATE)  # Compute tau_fix and tau_pay
    df_c = np.exp(-(r + s) * tp)  # Compute credit-adjusted discount factor
    fwd = S * np.exp((r - q) * tf)  # Compute forward price F_0 at final fixing
    v_tot = sig * np.sqrt(tf)  # Compute total volatility sigma * sqrt(tau_fix)
    f_m = lambda x, nd=4: fmt_num(x, nd, True)  # Math-mode number formatting helper
    d2_i = (np.log(fwd / INIT_FIX) - 0.5 * v_tot ** 2) / v_tot  # Compute d2(I) for binary bonus call
    d1_k = (np.log(fwd / STRIKE_BONUS) + 0.5 * v_tot ** 2) / v_tot  # Compute d1(K_b) for outperformance call
    d2_k = d1_k - v_tot  # Compute d2(K_b) for outperformance call
    bnd, dg, cl = non_callable_parts(S, r, q, s, sig, t0)  # Compute individual component PVs
    call_pv23 = EARLY_REDEMP_AMT * np.exp(-(r + s) * year_frac(T23, PAY23))  # Compute discounted call payout R * delta_23
    tau23 = year_frac(t0, T23)  # Compute year fraction to final observation date T23
    s_star = i["s_star"]  # Retrieve critical price threshold S* at T23
    fwd_23 = S * np.exp((r - q) * tau23)  # Compute forward price of TLT to T23
    d_star = (np.log(fwd_23 / s_star) - 0.5 * sig ** 2 * tau23) / (sig * np.sqrt(tau23))  # Compute d2(S*) at T23
    b = i["base"]  # Retrieve Scenario B baseline results
    add(r"\newcommand{\WorkedExample}{\begin{align*}")  # Start \WorkedExample align* macro for Appendix A
    add(f"\\tau_{{\\rm fix}}&=\\tfrac{{{(FINAL_FIX_DATE - t0).days}}}{{365}}={f_m(tf)},\\quad "  # Line 1: tau_fix,
        f"\\tau_{{\\rm pay}}=\\tfrac{{{(REDEMP_DATE - t0).days}}}{{365}}={f_m(tp)},\\quad N={i['N']}\\\\")  # tau_pay, and N
    add(f"\\sigma_{{\\text{{GARCH-TS}}}}&={f_m(100 * sig, 2)}\\,\\%,\\quad "  # Line 2: GARCH-TS vol,
        f"\\sigma\\sqrt{{\\tau_{{\\rm fix}}}}={f_m(v_tot)},\\quad DF={f_m(df_c, 5)},\\quad F_0={f_m(fwd, 4)}\\\\")  # total vol, DF, forward
    add(f"d_2(I)&={f_m(d2_i)}\\ (N={f_m(norm.cdf(d2_i))}),\\quad d_1(K_b)={f_m(d1_k)}\\ (N={f_m(norm.cdf(d1_k))}),\\quad d_2(K_b)={f_m(d2_k)}\\\\")  # Line 3: d1/d2
    add(f"V^{{\\rm NC}}&={f_m(bnd, 2)}+{f_m(dg, 2)}+{f_m(cl, 2)}={f_m(b['nc'], 2)}\\ \\text{{USD}}\\\\")  # Line 4: non-callable sum
    add(f"R\\,\\delta_{{23}}&={f_m(call_pv23, 2)},\\quad S^*={f_m(s_star, 2)},\\quad Q(S_{{t_{{23}}}}>S^*)={f_m(norm.cdf(d_star))}\\\\")  # Line 5: call params
    add(f"V^{{\\rm Cert}}&=V^{{\\rm NC}}-V^{{\\rm Berm}}={f_m(b['nc'], 2)}-{f_m(b['berm'], 2)}={f_m(b['cert'], 2)}\\ \\text{{USD}}\\ ({f_m(b['cert'] / 10.0, 2)}\\,\\%)")  # Line 6: final value
    add(r"\end{align*}}")  # Close \WorkedExample macro

    with open(out_path, "w", encoding="utf-8") as f_out:  # Open destination outputs.tex in UTF-8 write mode
        f_out.write("\n".join(lines) + "\n")  # Join all LaTeX lines with newlines and write file to disk


if __name__ == "__main__":  # Run main() only when executed directly as a script
    main()  # Invoke main valuation and export workflow
