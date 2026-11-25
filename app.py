import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
import io
import requests

st.set_page_config(page_title="German Football DC Predictor", page_icon="⚽", layout="wide")

# ---------------------------------------------------------
# 1. DATA LOADER (Bundesliga + 2. Bundesliga)
# ---------------------------------------------------------
@st.cache_data(ttl=86400)
def load_german_data():
    seasons = ["2425", "2526", "2627"]
    leagues = {"D1": "Bundesliga", "D2": "2. Bundesliga"}
    frames = []
    
    headers = {"User-Agent": "Mozilla/5.0"}
    for s in seasons:
        for code, name in leagues.items():
            url = f"https://football-data.co.uk{s}/{code}.csv"
            try:
                r = requests.get(url, headers=headers, timeout=10)
                if r.status_code == 200:
                    df = pd.read_csv(io.StringIO(r.text))
                    req_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG"]
                    if all(c in df.columns for c in req_cols):
                        df = df[req_cols].dropna()
                        df["League"] = name
                        df["Tier"] = 1 if code == "D1" else 2
                        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
                        df = df.dropna(subset=["Date"])
                        frames.append(df)
            except Exception:
                continue

    full_df = pd.concat(frames, ignore_index=True)
    # Calculate 2nd-half goals
    full_df["SHHG"] = (full_df["FTHG"] - full_df["HTHG"]).clip(lower=0)
    full_df["SHAG"] = (full_df["FTAG"] - full_df["HTAG"]).clip(lower=0)
    
    # Identify current season active teams per league
    latest_date = full_df["Date"].max()
    recent_df = full_df[full_df["Date"] >= (latest_date - pd.Timedelta(days=120))]
    d1_teams = sorted(recent_df[recent_df["Tier"] == 1]["HomeTeam"].unique())
    d2_teams = sorted(recent_df[recent_df["Tier"] == 2]["HomeTeam"].unique())
    
    return full_df, d1_teams, d2_teams

# ---------------------------------------------------------
# 2. DIXON-COLES ENGINE (With Tier Offset & Time Decay)
# ---------------------------------------------------------
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
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    
    max_date = df["Date"].max()
    days_ago = (max_date - df["Date"]).dt.days.values
    weights = np.exp(-xi * days_ago)
    
    h_idx = df["HomeTeam"].map(team_idx).values
    a_idx = df["AwayTeam"].map(team_idx).values
    h_goals = df[home_col].astype(int).values
    a_goals = df[away_col].astype(int).values
    is_d2 = (df["Tier"] == 2).astype(float).values

    # Parameters: [attack (n), defence (n), home_adv, rho, d2_offset]
    init_params = np.concatenate([
        np.zeros(n_teams),
        np.zeros(n_teams),
        [0.25, -0.08, -0.22]
    ])

    def log_likelihood(params):
        att = params[:n_teams]
        defn = params[n_teams:2*n_teams]
        home_adv = params[2*n_teams]
        rho = params[2*n_teams + 1]
        d2_off = params[2*n_teams + 2]

        # Scale ratings slightly in 2. Bundesliga to bridge promotion/relegation
        tier_adj = is_d2 * d2_off
        lam = np.exp(att[h_idx] + defn[a_idx] + home_adv + tier_adj)
        mu = np.exp(att[a_idx] + defn[h_idx] + tier_adj)

        # Vectorized tau
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

        ll = weights * (
            np.log(tau)
            + h_goals * np.log(lam) - lam
            + a_goals * np.log(mu) - mu
        )
        # Sum-to-zero constraint penalty for identifiability
        penalty = 100 * (np.sum(att) ** 2)
        return -np.sum(ll) + penalty

    bounds = [(-2.5, 2.5)] * (2 * n_teams) + [(0.0, 0.8), (-0.35, 0.1), (-0.6, 0.0)]
    res = minimize(log_likelihood, init_params, method="L-BFGS-B", bounds=bounds)
    
    p = res.x
    return {
        "teams": teams,
        "team_idx": team_idx,
        "att": p[:n_teams],
        "def": p[n_teams:2*n_teams],
        "home_adv": p[2*n_teams],
        "rho": p[2*n_teams + 1],
        "d2_off": p[2*n_teams + 2],
    }

def score_matrix(model, home_team, away_team, tier=1, max_goals=7):
    hi = model["team_idx"][home_team]
    ai = model["team_idx"][away_team]
    tier_adj = model["d2_off"] if tier == 2 else 0.0
    
    lam = np.exp(model["att"][hi] + model["def"][ai] + model["home_adv"] + tier_adj)
    mu = np.exp(model["att"][ai] + model["def"][hi] + tier_adj)
    rho = model["rho"]
    
    mat = np.zeros((max_goals, max_goals))
    for x in range(max_goals):
        for y in range(max_goals):
            t = dc_tau(x, y, lam, mu, rho)
            mat[x, y] = max(t, 0) * poisson.pmf(x, lam) * poisson.pmf(y, mu)
    mat /= np.sum(mat)
    return mat, lam, mu

# ---------------------------------------------------------
# 3. STREAMLIT USER INTERFACE
# ---------------------------------------------------------
st.title("🇩🇪 Bundesliga & 2. Bundesliga Score Predictor")
st.caption("Two-Half Dixon-Coles Model • Accounts for 1st/2nd Half Team Profiles, Time Decay & Tier Strength")

with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔄 Refresh Latest Match Data"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.write("**How it works:**")
    st.write("• Models **1st Half** and **2nd Half** attack/defence profiles separately.")
    st.write("• Combines both halves to calculate **FT**, **HT**, and **HT/FT** double odds.")

with st.spinner("Loading German league data & calibrating 1st/2nd half models..."):
    df, d1_teams, d2_teams = load_german_data()
    model_1h = fit_dixon_coles(df, "HTHG", "HTAG")
    model_2h = fit_dixon_coles(df, "SHHG", "SHAG")

# Competition selector
comp = st.radio(
    "Select Competition:",
    ["Bundesliga", "2. Bundesliga", "DFB-Pokal (All 36 Clubs)"],
    horizontal=True
)

if comp == "Bundesliga":
    team_list = d1_teams
    tier_val = 1
elif comp == "2. Bundesliga":
    team_list = d2_teams
    tier_val = 2
else:
    team_list = sorted(list(set(d1_teams + d2_teams)))
    tier_val = 1

col1, col2 = st.columns(2)
home_team = col1.selectbox("🏠 Home Team", team_list, index=0)
away_team = col2.selectbox("✈️ Away Team", team_list, index=min(1, len(team_list)-1))

if st.button("⚽ Predict Match Outcomes", type="primary", use_container_width=True):
    if home_team == away_team:
        st.error("Please select two different teams.")
    else:
        # 1st Half Matrix & 2nd Half Matrix
        mat_1h, lam_1h, mu_1h = score_matrix(model_1h, home_team, away_team, tier=tier_val, max_goals=6)
        mat_2h, lam_2h, mu_2h = score_matrix(model_2h, home_team, away_team, tier=tier_val, max_goals=6)
        
        # Convolve 1H and 2H matrices to get exact Full-Time & HT/FT combinations
        ft_mat = np.zeros((10, 10))
        ht_ft_probs = {
            "Home / Home": 0.0, "Home / Draw": 0.0, "Home / Away": 0.0,
            "Draw / Home": 0.0, "Draw / Draw": 0.0, "Draw / Away": 0.0,
            "Away / Home": 0.0, "Away / Draw": 0.0, "Away / Away": 0.0,
        }
        
        for h1 in range(6):
            for a1 in range(6):
                p1 = mat_1h[h1, a1]
                if p1 < 1e-7:
                    continue
                ht_res = "Home" if h1 > a1 else ("Away" if a1 > h1 else "Draw")
                
                for h2 in range(6):
                    for a2 in range(6):
                        p2 = mat_2h[h2, a2]
                        prob = p1 * p2
                        fh, fa = h1 + h2, a1 + a2
                        if fh < 10 and fa < 10:
                            ft_mat[fh, fa] += prob
                        ft_res = "Home" if fh > fa else ("Away" if fa > fh else "Draw")
                        ht_ft_probs[f"{ht_res} / {ft_res}"] += prob

        ft_mat /= np.sum(ft_mat)
        
        # Key Metrics
        ht_home = np.sum(np.tril(mat_1h, -1))
        ht_draw = np.sum(np.trace(mat_1h))
        ht_away = np.sum(np.triu(mat_1h, 1))
        
        ft_home = np.sum(np.tril(ft_mat, -1))
        ft_draw = np.sum(np.trace(ft_mat))
        ft_away = np.sum(np.triu(ft_mat, 1))

        best_ht = np.unravel_index(np.argmax(mat_1h), mat_1h.shape)
        best_ft = np.unravel_index(np.argmax(ft_mat), ft_mat.shape)

        st.markdown("---")
        st.subheader(f"📊 {home_team} vs {away_team}")

        # Top Banner: Most Likely Scores
        m1, m2, m3 = st.columns(3)
        m1.metric("Most Likely Full-Time Score", f"{best_ft} - {best_ft}", f"{ft_mat[best_ft]:.1%} chance")
        m2.metric("Most Likely Half-Time Score", f"{best_ht} - {best_ht}", f"{mat_1h[best_ht]:.1%} chance")
        m3.metric("Expected Goals Split (H - A)", f"{lam_1h+lam_2h:.2f} - {mu_1h+mu_2h:.2f}", f"1H: {lam_1h:.2f}-{mu_1h:.2f} | 2H: {lam_2h:.2f}-{mu_2h:.2f}")

        # Full Time & Half Time 1X2
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### ⏱️ Full-Time Result (90 Mins)")
            ft_df = pd.DataFrame({
                "Outcome": [f"{home_team} Win", "Draw", f"{away_team} Win"],
"Probability": [f"{ft_home:.1%}", f"{ft_draw:.1%}", f"{ft_away:.1%}"],
"Fair Odds": [f"{1/max(ft_home,0.001):.2f}", f"{1/max(ft_draw,0.001):.2f}", f"{1/max(ft_away,0.001):.2f}"]
})
st.dataframe(ft_df, hide_index=True, use_container_width=True)
with c2:
st.markdown("#### ⏸️ Half-Time Result (45 Mins)")
ht_df = pd.DataFrame({
"Outcome": [f"{home_team} Leading", "Level at HT", f"{away_team} Leading"],
"Probability": [f"{ht_home:.1%}", f"{ht_draw:.1%}", f"{ht_away:.1%}"],
"Fair Odds": [f"{1/max(ht_home,0.001):.2f}", f"{1/max(ht_draw,0.001):.2f}", f"{1/max(ht_away,0.001):.2f}"]
})
st.dataframe(ht_df, hide_index=True, use_container_width=True)
# Top Exact Scorelines & HT/FT Combos
c3, c4 = st.columns(2)
with c3:
st.markdown("#### 🎯 Top 6 Most Likely FT Scorelines")
flat_indices = np.argsort(ft_mat.ravel())[::-1][:6]
score_rows = []
for idx in flat_indices:
r, c = divmod(idx, 10)
p = ft_mat[r, c]
score_rows.append({
"Scoreline": f"{r} - {c}",
"Probability": f"{p:.1%}",
"Fair Odds": f"{1/max(p,0.001):.2f}"
})
st.dataframe(pd.DataFrame(score_rows), hide_index=True, use_container_width=True)
with c4:
st.markdown("#### 🔀 Half-Time / Full-Time (All 9 Combos)")
sorted_htft = sorted(ht_ft_probs.items(), key=lambda x: x[1], reverse=True)
htft_rows = []
for label, p in sorted_htft:
htft_rows.append({
"HT / FT": label,
"Probability": f"{p:.1%}",
"Fair Odds": f"{1/max(p,0.001):.2f}"
})
st.dataframe(pd.DataFrame(htft_rows), hide_index=True, use_container_width=True)
