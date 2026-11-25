import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
import urllib.request
import ssl
import io

@st.cache_data(ttl=86400)
def load_german_data(cache_ver=4):
    seasons = ["2425", "2526", "2627"]
    leagues = [("D1", "Bundesliga", 1), ("D2", "2. Bundesliga", 2)]
    frames = []
    errors = []

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for s in seasons:
        for code, name, tier in leagues:
            url = f"https://football-data.co.uk{s}/{code}.csv"
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                )
                with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                    raw_text = resp.read().decode("latin1")
                df = pd.read_csv(io.StringIO(raw_text))
                req_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG"]
                if all(c in df.columns for c in req_cols):
                    df = df[req_cols].dropna()
                    df["League"] = name
                    df["Tier"] = tier
                    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
                    df = df.dropna(subset=["Date"])
                    frames.append(df)
            except Exception as e:
                errors.append(f"{s}/{code}: {e}")

    if not frames:
        st.error(f"Data fetch failed. Details: {errors[:2]}")
        st.stop()

    full_df = pd.concat(frames, ignore_index=True)
    full_df["SHHG"] = (full_df["FTHG"] - full_df["HTHG"]).clip(lower=0)
    full_df["SHAG"] = (full_df["FTAG"] - full_df["HTAG"]).clip(lower=0)

    latest_date = full_df["Date"].max()
    recent_df = full_df[full_df["Date"] >= (latest_date - pd.Timedelta(days=120))]
    d1_teams = sorted(recent_df[recent_df["Tier"] == 1]["HomeTeam"].unique())
    d2_teams = sorted(recent_df[recent_df["Tier"] == 2]["HomeTeam"].unique())
    return full_df, d1_teams, d2_teams

def dc_tau(x, y, lambda_x, mu_y, rho):
    if x == 0 and y == 0:
        return 1 - (lambda_x * mu_y * rho)
    elif x == 0 and y == 1:
        return 1 + (lambda_x * rho)
    elif x == 1 and y == 0:
        return 1 + (mu_y * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0

def fit_dixon_coles(df, home_col, away_col, xi=0.0022):
    teams = sorted(list(set(df["HomeTeam"]).union(set(df["AwayTeam"]))))
    team_idx = dict(zip(teams, range(len(teams))))
    n_teams = len(teams)

    max_date = df["Date"].max()
    days_ago = (max_date - df["Date"]).dt.days.values
    weights = np.exp(-xi * days_ago)

    h_idx = df["HomeTeam"].map(team_idx).values
    a_idx = df["AwayTeam"].map(team_idx).values
    h_goals = df[home_col].astype(int).values
    a_goals = df[away_col].astype(int).values
    is_d2 = (df["Tier"] == 2).astype(float).values

    init_params = np.concatenate([np.zeros(n_teams), np.zeros(n_teams), [0.25, -0.08, -0.22]])

    def log_likelihood(params):
        att = params[:n_teams]
        defn = params[n_teams:2*n_teams]
        home_adv = params[2*n_teams]
        rho = params[2*n_teams + 1]
        d2_off = params[2*n_teams + 2]

        tier_adj = is_d2 * d2_off
        lam = np.exp(att[h_idx] + defn[a_idx] + home_adv + tier_adj)
        mu = np.exp(att[a_idx] + defn[h_idx] + tier_adj)

        tau = np.ones_like(lam)
        m00 = (h_goals == 0) & (a_goals == 0)
        m01 = (h_goals == 0) & (a_goals == 1)
        m10 = (h_goals == 1) & (a_goals == 0)
        m11 = (h_goals == 1) & (a_goals == 1)

        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m10] = 1 + mu[m10] * rho
        tau[m11] = 1 - rho
        tau = np.maximum(tau, 1e-6)

        ll = weights * (np.log(tau) + h_goals * np.log(lam) - lam + a_goals * np.log(mu) - mu)
        penalty = 100 * (np.sum(att) ** 2)
        return -np.sum(ll) + penalty

    bounds = [(-2.5, 2.5)] * (2 * n_teams) + [(0.0, 0.8), (-0.35, 0.1), (-0.6, 0.0)]
    res = minimize(log_likelihood, init_params, method="L-BFGS-B", bounds=bounds)
    p = res.x
    return (teams, team_idx, p[:n_teams], p[n_teams:2*n_teams], p[2*n_teams], p[2*n_teams + 1], p[2*n_teams + 2])

def score_matrix(model, home_team, away_team, tier=1, max_goals=7):
    teams, team_idx, att, defn, home_adv, rho, d2_off = model
    hi = team_idx[home_team]
    ai = team_idx[away_team]
    tier_adj = d2_off if tier == 2 else 0.0

    lam = np.exp(att[hi] + defn[ai] + home_adv + tier_adj)
    mu = np.exp(att[ai] + defn[hi] + tier_adj)

    mat = np.zeros((max_goals, max_goals))
    for x in range(max_goals):
        for y in range(max_goals):
            t = dc_tau(x, y, lam, mu, rho)
            mat[x, y] = max(t, 0) * poisson.pmf(x, lam) * poisson.pmf(y, mu)
    mat /= np.sum(mat)
    return mat, lam, mu

# END OF ENGINE
