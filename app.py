import streamlit as st
import pandas as pd
import numpy as np
from engine import load_german_data, fit_dixon_coles, score_matrix

st.set_page_config(page_title="German Football DC Predictor", page_icon="⚽", layout="wide")

st.title("🇩🇪 Bundesliga & 2. Bundesliga Score Predictor")
st.caption("Shot-Proxy xG Blended (60/40) • Two-Half Dixon-Coles Model • Tier & Time-Decay Adjusted")

with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔄 Refresh Latest Match Data"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.write("**Model Features:**")
    st.write("• **60/40 xG Blend:** Combines Shot-Proxy xG with actual goals to filter finishing luck.")
    st.write("• **Two-Half Grids:** Separate 1st-half and 2nd-half ratings for HT, FT & HT/FT odds.")

with st.spinner("Loading German league data & calibrating xG-blended models..."):
    df, d1_teams, d2_teams = load_german_data()
    model_1h = fit_dixon_coles(df, "BLEND_1H_H", "BLEND_1H_A", "HTHG", "HTAG")
    model_2h = fit_dixon_coles(df, "BLEND_2H_H", "BLEND_2H_A", "SHHG", "SHAG")

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
        mat_1h, lam_1h, mu_1h = score_matrix(model_1h, home_team, away_team, tier=tier_val, max_goals=6)
        mat_2h, lam_2h, mu_2h = score_matrix(model_2h, home_team, away_team, tier=tier_val, max_goals=6)

        ft_mat = np.zeros((10, 10))
        labels = [
            "Home / Home", "Home / Draw", "Home / Away",
            "Draw / Home", "Draw / Draw", "Draw / Away",
            "Away / Home", "Away / Draw", "Away / Away"
        ]
        ht_ft_probs = dict(zip(labels, [0.0] * 9))

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

        ht_home = np.sum(np.tril(mat_1h, -1))
        ht_draw = np.sum(np.trace(mat_1h))
        ht_away = np.sum(np.triu(mat_1h, 1))

        ft_home = np.sum(np.tril(ft_mat, -1))
        ft_draw = np.sum(np.trace(ft_mat))
        ft_away = np.sum(np.triu(ft_mat, 1))

        # Over/Under 2.5 Goals & BTTS probabilities
        over_25 = sum(ft_mat[r, c] for r in range(10) for c in range(10) if (r + c) >= 3)
        under_25 = 1.0 - over_25
        btts_yes = sum(ft_mat[r, c] for r in range(1, 10) for c in range(1, 10))
        btts_no = 1.0 - btts_yes

        best_ht = np.unravel_index(np.argmax(mat_1h), mat_1h.shape)
        best_ft = np.unravel_index(np.argmax(ft_mat), ft_mat.shape)
        ht_h, ht_a = best_ht
        ft_h, ft_a = best_ft

        st.markdown("---")
        st.subheader(f"📊 {home_team} vs {away_team}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Most Likely Full-Time Score", f"{int(ft_h)} - {int(ft_a)}", f"{ft_mat[best_ft]:.1%} chance")
        m2.metric("Most Likely Half-Time Score", f"{int(ht_h)} - {int(ht_a)}", f"{mat_1h[best_ht]:.1%} chance")
        m3.metric("Blended xG Split (H - A)", f"{lam_1h+lam_2h:.2f} - {mu_1h+mu_2h:.2f}", f"1H: {lam_1h:.2f}-{mu_1h:.2f} | 2H: {lam_2h:.2f}-{mu_2h:.2f}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### ⏱️ Full-Time Result (90 Mins)")
            ft_rows = [
                (f"{home_team} Win", f"{ft_home:.1%}", f"{1/max(ft_home,0.001):.2f}"),
                ("Draw", f"{ft_draw:.1%}", f"{1/max(ft_draw,0.001):.2f}"),
                (f"{away_team} Win", f"{ft_away:.1%}", f"{1/max(ft_away,0.001):.2f}")
            ]
            st.dataframe(pd.DataFrame(ft_rows, columns=["Outcome", "Probability", "Fair Odds"]), hide_index=True, use_container_width=True)

        with c2:
            st.markdown("#### ⏸️ Half-Time Result (45 Mins)")
            ht_rows = [
                (f"{home_team} Leading", f"{ht_home:.1%}", f"{1/max(ht_home,0.001):.2f}"),
                ("Level at HT", f"{ht_draw:.1%}", f"{1/max(ht_draw,0.001):.2f}"),
                (f"{away_team} Leading", f"{ht_away:.1%}", f"{1/max(ht_away,0.001):.2f}")
            ]
            st.dataframe(pd.DataFrame(ht_rows, columns=["Outcome", "Probability", "Fair Odds"]), hide_index=True, use_container_width=True)

        with c3:
            st.markdown("#### 🔥 Goals & BTTS Markets")
            goal_rows = [
                ("Over 2.5 Goals", f"{over_25:.1%}", f"{1/max(over_25,0.001):.2f}"),
                ("Under 2.5 Goals", f"{under_25:.1%}", f"{1/max(under_25,0.001):.2f}"),
                ("BTTS - Yes", f"{btts_yes:.1%}", f"{1/max(btts_yes,0.001):.2f}"),
                ("BTTS - No", f"{btts_no:.1%}", f"{1/max(btts_no,0.001):.2f}")
            ]
            st.dataframe(pd.DataFrame(goal_rows, columns=["Market", "Probability", "Fair Odds"]), hide_index=True, use_container_width=True)

        c4, c5 = st.columns(2)
        with c4:
            st.markdown("#### 🎯 Top 6 Most Likely FT Scorelines")
            flat_indices = np.argsort(ft_mat.ravel())[::-1][:6]
            score_rows = []
            for idx in flat_indices:
                r, c = divmod(idx, 10)
                p = ft_mat[r, c]
                score_rows.append((f"{int(r)} - {int(c)}", f"{p:.1%}", f"{1/max(p,0.001):.2f}"))
            st.dataframe(pd.DataFrame(score_rows, columns=["Scoreline", "Probability", "Fair Odds"]), hide_index=True, use_container_width=True)

        with c5:
            st.markdown("#### 🔀 Half-Time / Full-Time (All 9 Combos)")
            ranked_htft = sorted([(p, label) for label, p in ht_ft_probs.items()], reverse=True)
            htft_rows = []
            for p, label in ranked_htft:
                htft_rows.append((label, f"{p:.1%}", f"{1/max(p,0.001):.2f}"))
            st.dataframe(pd.DataFrame(htft_rows, columns=["HT / FT", "Probability", "Fair Odds"]), hide_index=True, use_container_width=True)

# END OF APP
