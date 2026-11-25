import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.special import gammaln
import glob
import os

@st.cache_data(ttl=86400)
def load_german_data(cache_ver=6):
    csv_files = glob.glob("*.csv")
    frames = []

    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="latin1")
            req_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG"]
            if all(c in df.columns for c in req_cols):
                keep = req_cols + [c for c in ["HS", "AS", "HST", "AST"] if c in df.columns]
                df = df[keep].dropna()
                fname = os.path.basename(f).upper()
                df["Tier"] = 2 if "D2" in fname else 1
                df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
                df = df.dropna(subset=["Date"])
                frames.append(df)
        except Exception:
            continue

    if not frames:
        st.error("No CSV files found in repository!")
        st.stop()

    full_df = pd.concat(frames, ignore_index=True)
    full_df["SHHG"] = (full_df["FTHG"] - full_df["HTHG"]).clip(lower=0)
    full_df["SHAG"] = (full_df["FTAG"] - full_df["HTAG"]).clip(lower=0)

    # Calculate Shot-Proxy xG & blend 60% Proxy xG / 40% Actual Goals
    if all(c in full_df.columns for c in ["HS", "AS", "HST", "AST"]):
        h_off = (full_df["HS"] - full_df["HST"]).clip(lower=0)
        a_off = (full_df["AS"] - full_df["AST"]).clip(lower=0)
        ft_xg_h = 0.31 * full_df["HST"] + 0.05 * h_off
        ft_xg_a = 0.31 * full_df["AST"] + 0.05 * a_off

        # Split proxy xG into 1H and 2H using team half-goal tendencies (default 44% 1H / 56% 2H)
        h_ratio = np.where(full_df["FTHG"] > 0, 0.5 * (full_df["HTHG"] / full_df["FTHG"]) + 0.5 * 0.44, 0.44)
        a_ratio = np.where(full_df["FTAG"] > 0, 0.5 * (full_df["HTAG"] / full_df["FTAG"]) + 0.5 * 0.44, 0.44)

        full_df["BLEND_1H_H"] = 0.60 * (ft_xg_h * h_ratio) + 0.40 * full_df["HTHG"]
        full_df["BLEND_1H_A"] = 0.60 * (ft_xg_a * a_ratio) + 0.40 * full_df["HTAG"]
        full_df["BLEND_2H_H"] = 0.60 * (ft_xg_h * (1 - h_ratio)) + 0.40 * full_df["SHHG"]
        full_df["BLEND_2H_A"] = 0.60 * (ft_xg_a * (1 - a_ratio)) + 0.40 * full_df["SHAG"]
    else:
        full_df["BLEND_1H_H"] = full_df["HTHG"].astype(float)
        full_df["BLEND_1H_A"] = full_df["HTAG"].astype(float)
        full_df["BLEND_2H_H"] = full_df["SHHG"].astype(float)
        full_df["BLEND_2H_A"] = full_df["SHAG"].astype(float)

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

def fit_dixon_coles(df, home_col, away_col, raw_h_col, raw_a_col, xi=0.0022):
    teams = sorted(list(set(df["HomeTeam"]).union(set(df["AwayTeam"]))))
    team_idx = dict(zip(teams, range(len(teams))))
    n_teams = len(teams)

    max_date = df["Date"].max()
    days_ago = (max_date - df["Date"]).dt.days.values
    weights = np.exp(-xi * days_ago)

    h_idx = df["HomeTeam"].map(team_idx).values
    a_idx = df["AwayTeam"].map(team_idx).values
    h_val = df[home_col].astype(float).values
    a_val = df[away_col].astype(float).values
    raw_h = df[raw_h_col].astype(int).values
    raw_a = df[raw_a_col].astype(int).values
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
        m00 = (raw_h == 0) & (raw_a == 0)
        m01 = (raw_h == 0) & (raw_a == 1)
        m10 = (raw_h == 1) & (raw_a == 0)
        m11 = (raw_h == 1) & (raw_a == 1)

        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m10] = 1 + mu[m10] * rho
        tau[m11] = 1 - rho
        tau = np.maximum(tau, 1e-6)

        ll = weights * (
            np.log(tau)
            + h_val * np.log(lam) - lam - gammaln(h_val + 1)
            + a_val * np.log(mu) - mu - gammaln(a_val + 1)
        )
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
