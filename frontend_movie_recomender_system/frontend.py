import streamlit as st
import pickle
import pandas as pd
import ast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

# ── Session State ─────────────────────────────────────────────────────────────
if "detail_title"    not in st.session_state: st.session_state.detail_title    = None
if "recommendations" not in st.session_state: st.session_state.recommendations = []
if "dark_mode"       not in st.session_state: st.session_state.dark_mode       = True

# ── Theme ─────────────────────────────────────────────────────────────────────
D = st.session_state.dark_mode
BG       = "#07071a" if D else "#f0f2f8"
CARD     = "#12122a" if D else "#ffffff"
CARD2    = "#1a1a35" if D else "#eef0f8"
BORDER   = "#2a2a50" if D else "#d0d4e8"
TEXT     = "#f0f0ff" if D else "#0a0a20"
SUBTEXT  = "#7777aa" if D else "#555577"
ACCENT   = "#e94560"
ACCENT2  = "#7b61ff"
ACCENT3  = "#00d4aa"
TOGGLE   = "☀️ Light" if D else "🌙 Dark"
PLOT_BG  = "#0d0d20" if D else "#ffffff"
PLOT_PAP = "#07071a" if D else "#f0f2f8"
FONT_COL = "#f0f0ff" if D else "#0a0a20"
GRID_COL = "#1e1e40" if D else "#e0e0f0"

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

[data-testid="collapsedControl"] {{ display:none !important; }}
section[data-testid="stSidebar"] {{ display:none !important; }}

html, body, .stApp {{
    background-color: {BG} !important;
    font-family: 'Inter', sans-serif !important;
}}
.block-container {{ padding-top: 0 !important; max-width: 1300px; }}

/* All text */
p, span, div, label, h1, h2, h3, h4, li {{ color: {TEXT} !important; }}
.stCaption, .stCaption p {{ color: {SUBTEXT} !important; font-size: 13px !important; }}

/* Tabs */
[data-baseweb="tab-list"] {{
    background: {CARD} !important;
    border-radius: 14px !important;
    padding: 6px !important;
    gap: 4px !important;
    border: 1px solid {BORDER} !important;
}}
[data-baseweb="tab"] {{
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    color: {SUBTEXT} !important;
    padding: 8px 20px !important;
    border: none !important;
    background: transparent !important;
}}
[aria-selected="true"] {{
    background: linear-gradient(135deg, {ACCENT}, {ACCENT2}) !important;
    color: white !important;
}}

/* Buttons */
div.stButton > button {{
    height: 48px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    border-radius: 12px !important;
    width: 100% !important;
    border: none !important;
    background: linear-gradient(135deg, {ACCENT}, {ACCENT2}) !important;
    color: white !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(233,69,96,0.3) !important;
}}
div.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(233,69,96,0.4) !important;
}}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {{
    background: {CARD} !important;
    border: 1.5px solid {BORDER} !important;
    border-radius: 12px !important;
    color: {TEXT} !important;
    font-size: 15px !important;
}}

/* Metrics */
[data-testid="stMetric"] {{
    background: {CARD} !important;
    border-radius: 14px !important;
    padding: 16px !important;
    border: 1px solid {BORDER} !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important;
}}
[data-testid="stMetricValue"] {{ font-size: 22px !important; font-weight: 800 !important; }}

/* Progress bar */
[data-testid="stProgressBar"] > div {{ background: {BORDER} !important; border-radius: 99px !important; }}
[data-testid="stProgressBar"] > div > div {{
    background: linear-gradient(90deg, {ACCENT}, {ACCENT2}) !important;
    border-radius: 99px !important;
}}

/* Info box */
[data-testid="stAlert"] {{
    background: {CARD2} !important;
    border: 1px solid {BORDER} !important;
    border-left: 4px solid {ACCENT} !important;
    border-radius: 12px !important;
    color: {TEXT} !important;
}}
[data-testid="stAlert"] p {{ color: {TEXT} !important; }}

/* Container borders */
[data-testid="stVerticalBlockBorderWrapper"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 16px !important;
    background: {CARD} !important;
    padding: 4px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.15) !important;
}}

/* Expander */
[data-testid="stExpander"] {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
}}

hr {{ border-color: {BORDER} !important; opacity: 0.5; }}
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    import os
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(BASE_DIR)
    movies_dict = pickle.load(open(os.path.join(BASE_DIR, "movies_dict.pkl"), "rb"))
    movies      = pd.DataFrame(movies_dict)
    similarity  = pickle.load(open(os.path.join(BASE_DIR, "similarity.pkl"), "rb"))
    movies_full = pd.read_csv(os.path.join(BASE_DIR, "tmdb_5000_movies.csv"))
    credits     = pd.read_csv(os.path.join(BASE_DIR, "tmdb_5000_credits.csv"))
    movies_full = movies_full.merge(credits, on="title")
    return movies, similarity, movies_full

movies, similarity, movies_full = load_data()

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_genres(s):
    try:    return [g["name"] for g in ast.literal_eval(s)]
    except: return []

def get_cast(s, n=5):
    try:    return [c["name"] for c in ast.literal_eval(s)[:n]]
    except: return []

def get_director(s):
    try:
        for c in ast.literal_eval(s):
            if c["job"] == "Director": return c["name"]
    except: pass
    return "N/A"

def get_details(title):
    row = movies_full[movies_full["title"] == title]
    if row.empty: return None
    r = row.iloc[0]
    rel = str(r["release_date"])
    return {
        "title":    title,
        "tagline":  str(r["tagline"])  if str(r["tagline"])  != "nan" else "",
        "overview": str(r["overview"]) if str(r["overview"]) != "nan" else "No overview.",
        "rating":   round(float(r["vote_average"]), 1),
        "votes":    int(r["vote_count"]),
        "year":     rel[:4] if rel != "nan" else "N/A",
        "runtime":  int(r["runtime"])  if str(r["runtime"])  not in ["nan","0"] else 0,
        "genres":   get_genres(r["genres"]),
        "language": str(r["original_language"]).upper(),
        "budget":   int(r["budget"])   if str(r["budget"])   != "nan" else 0,
        "revenue":  int(r["revenue"])  if str(r["revenue"])  != "nan" else 0,
        "cast":     get_cast(r["cast"]),
        "director": get_director(r["crew"]),
    }

OMDB_KEY  = "f67cfb9c"
PLACEHOLDER = "https://via.placeholder.com/300x450/12122a/7777aa?text=No+Poster"

@st.cache_data(show_spinner=False)
def fetch_poster(title):
    try:
        url  = f"https://www.omdbapi.com/?t={requests.utils.quote(title)}&apikey={OMDB_KEY}"
        data = requests.get(url, timeout=5).json()
        poster = data.get("Poster","")
        return poster if poster and poster != "N/A" else PLACEHOLDER
    except:
        return PLACEHOLDER

def get_movie_id(title):
    row = movies[movies["title"] == title]
    return int(row.iloc[0].movie_id) if not row.empty else None

def recommend(movie):
    idx  = movies[movies["title"] == movie].index[0]
    dist = similarity[idx]
    top  = sorted(enumerate(dist), key=lambda x: x[1], reverse=True)[1:6]
    return [movies.iloc[i].title for i, _ in top]

genre_emojis = {
    "Action":"💥","Adventure":"🗺️","Animation":"🎨","Comedy":"😂",
    "Crime":"🕵️","Documentary":"📽️","Drama":"🎭","Family":"👨‍👩‍👧",
    "Fantasy":"🧙","History":"📜","Horror":"😱","Music":"🎵",
    "Mystery":"🔍","Romance":"❤️","Science Fiction":"🚀",
    "Thriller":"😰","TV Movie":"📺","War":"⚔️","Western":"🤠"
}

# ── Details Card ──────────────────────────────────────────────────────────────
def show_details(title):
    d = get_details(title)
    if not d:
        st.warning("Details not found.")
        return

    # Header card
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{CARD2},{CARD});
                border:1px solid {BORDER}; border-left:5px solid {ACCENT};
                border-radius:20px; padding:28px 32px; margin:20px 0;
                box-shadow:0 8px 32px rgba(0,0,0,0.3);">
        <div style="font-size:28px;font-weight:900;letter-spacing:-0.5px;color:{TEXT};">
            🎬 {d['title']}
        </div>
        <div style="color:{SUBTEXT};font-style:italic;margin-top:6px;font-size:15px;">
            {'&quot;' + d['tagline'] + '&quot;' if d['tagline'] else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        # Show poster
        poster = fetch_poster(title)
        st.image(poster, use_container_width=True)
        rating = d["rating"]
        pct    = int(rating * 10)
        stars  = "⭐" * int(rating/2) + ("☆" * (5 - int(rating/2)))
        st.markdown(f"""
        <div style="background:linear-gradient(160deg,{CARD2},{CARD});
                    border:1px solid {BORDER}; border-radius:18px;
                    padding:24px; text-align:center; margin-bottom:14px;
                    box-shadow:0 4px 20px rgba(0,0,0,0.2);">
            <div style="font-size:11px;color:{SUBTEXT};letter-spacing:3px;margin-bottom:10px;">AUDIENCE SCORE</div>
            <div style="font-size:60px;font-weight:900;color:#f5c518;line-height:1;">{rating}</div>
            <div style="color:{SUBTEXT};font-size:11px;margin:4px 0;">OUT OF 10</div>
            <div style="font-size:20px;margin:8px 0;">{stars}</div>
            <div style="background:{BORDER};border-radius:99px;height:6px;margin:10px 4px 6px;">
                <div style="background:linear-gradient(90deg,{ACCENT},{ACCENT2});width:{pct}%;height:6px;border-radius:99px;"></div>
            </div>
            <div style="color:{SUBTEXT};font-size:12px;">{d['votes']:,} ratings</div>
        </div>
        """, unsafe_allow_html=True)

        runtime = f"{d['runtime']} min" if d["runtime"] else "N/A"
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:16px;padding:20px;">
            <div style="font-size:11px;color:{SUBTEXT};letter-spacing:2px;margin-bottom:14px;">MOVIE INFO</div>
            <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid {BORDER};">
                <span style="color:{SUBTEXT};font-size:13px;">📅 Year</span>
                <span style="font-weight:700;font-size:13px;color:{TEXT};">{d['year']}</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid {BORDER};">
                <span style="color:{SUBTEXT};font-size:13px;">⏱ Runtime</span>
                <span style="font-weight:700;font-size:13px;color:{TEXT};">{runtime}</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid {BORDER};">
                <span style="color:{SUBTEXT};font-size:13px;">🌐 Language</span>
                <span style="font-weight:700;font-size:13px;color:{TEXT};">{d['language']}</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:8px 0;">
                <span style="color:{SUBTEXT};font-size:13px;">🎬 Director</span>
                <span style="font-weight:700;font-size:13px;color:{TEXT};">{d['director']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if d["genres"]:
            genre_html = "".join([
                f"<span style='background:linear-gradient(135deg,{ACCENT},{ACCENT2});color:white;"
                f"padding:5px 14px;border-radius:20px;font-size:12px;font-weight:700;"
                f"margin:3px;display:inline-block;'>{g}</span>"
                for g in d["genres"]
            ])
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:16px;
                        padding:18px 20px;margin-bottom:14px;">
                <div style="font-size:11px;color:{SUBTEXT};letter-spacing:2px;margin-bottom:10px;">GENRES</div>
                {genre_html}
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-left:4px solid {ACCENT};
                    border-radius:16px;padding:20px;margin-bottom:14px;">
            <div style="font-size:11px;color:{SUBTEXT};letter-spacing:2px;margin-bottom:10px;">PLOT OVERVIEW</div>
            <div style="font-size:15px;line-height:1.8;color:{TEXT};">{d['overview']}</div>
        </div>
        """, unsafe_allow_html=True)

        if d["cast"]:
            cast_html = "".join([
                f"<span style='background:{CARD2};color:{ACCENT3};border:1px solid {BORDER};"
                f"padding:6px 14px;border-radius:20px;font-size:12px;font-weight:600;"
                f"margin:3px;display:inline-block;'>🌟 {c}</span>"
                for c in d["cast"]
            ])
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:16px;
                        padding:18px 20px;margin-bottom:14px;">
                <div style="font-size:11px;color:{SUBTEXT};letter-spacing:2px;margin-bottom:10px;">TOP CAST</div>
                {cast_html}
            </div>
            """, unsafe_allow_html=True)

        if d["budget"] > 0 or d["revenue"] > 0:
            m1, m2 = st.columns(2)
            with m1:
                st.metric("💰 Budget", f"${d['budget']:,}" if d["budget"] > 0 else "N/A")
            with m2:
                profit = d["revenue"] - d["budget"]
                st.metric("🏦 Box Office", f"${d['revenue']:,}" if d["revenue"] > 0 else "N/A",
                          delta=f"+${profit:,}" if profit > 0 and d["budget"] > 0 else None)

# ── Genre data ────────────────────────────────────────────────────────────────
@st.cache_data
def get_all_genres():
    s = set()
    for g in movies_full["genres"]:
        for item in get_genres(g): s.add(item)
    return sorted(list(s))

@st.cache_data
def get_top_by_genre(genre, n=10):
    mask     = movies_full["genres"].apply(lambda g: genre in get_genres(g))
    filtered = movies_full[mask & (movies_full["vote_count"] >= 100)].copy()
    return filtered.sort_values("vote_average", ascending=False).head(n)[
        ["title","vote_average","vote_count","release_date","genres"]
    ].reset_index(drop=True)

all_genres = get_all_genres()

# ══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="background:linear-gradient(135deg,#07071a 0%,#1a0a2e 40%,#0a1a3a 70%,#07071a 100%);border-bottom:1px solid {BORDER};padding:56px 48px 44px;margin-bottom:0;text-align:center;position:relative;overflow:hidden;">
<div style="position:absolute;top:-60px;left:20%;width:300px;height:300px;background:radial-gradient(circle,rgba(233,69,96,0.18),transparent 70%);border-radius:50%;pointer-events:none;"></div>
<div style="position:absolute;top:-40px;right:20%;width:280px;height:280px;background:radial-gradient(circle,rgba(123,97,255,0.18),transparent 70%);border-radius:50%;pointer-events:none;"></div>
<div style="display:inline-block;background:rgba(233,69,96,0.12);border:1px solid rgba(233,69,96,0.35);color:{ACCENT};font-size:11px;font-weight:700;letter-spacing:4px;text-transform:uppercase;padding:6px 20px;border-radius:30px;margin-bottom:20px;">✦ AI-Powered Movie Discovery ✦</div>
<div style="font-size:72px;font-weight:900;letter-spacing:-3px;line-height:1;background:linear-gradient(90deg,{ACCENT} 0%,{ACCENT2} 50%,{ACCENT3} 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:16px;">🎬 CineMatch</div>
<div style="color:{SUBTEXT};font-size:17px;font-weight:400;letter-spacing:0.5px;margin-bottom:28px;max-width:520px;margin-left:auto;margin-right:auto;line-height:1.6;">Tell us one movie you love.<br>We'll find five more you'll obsess over.</div>
<div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;">
<span style="background:rgba(233,69,96,0.12);color:{ACCENT};border:1px solid rgba(233,69,96,0.3);padding:7px 18px;border-radius:30px;font-size:12px;font-weight:700;">🎯 ML Powered</span>
<span style="background:rgba(123,97,255,0.12);color:{ACCENT2};border:1px solid rgba(123,97,255,0.3);padding:7px 18px;border-radius:30px;font-size:12px;font-weight:700;">⚡ Instant Results</span>
<span style="background:rgba(0,212,170,0.12);color:{ACCENT3};border:1px solid rgba(0,212,170,0.3);padding:7px 18px;border-radius:30px;font-size:12px;font-weight:700;">📊 5,000+ Movies</span>
<span style="background:rgba(255,200,0,0.10);color:#f5c518;border:1px solid rgba(255,200,0,0.25);padding:7px 18px;border-radius:30px;font-size:12px;font-weight:700;">⭐ TMDB Dataset</span>
</div>
</div>
""", unsafe_allow_html=True)

# Toggle button row
tc1, tc2 = st.columns([6,1])
with tc2:
    st.write("")
    if st.button(TOGGLE, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🍿  Recommender",
    "🏆  Top Rated by Genre",
    "📊  Statistics",
    "⚖️  Compare Movies"
])

# ══════════════════════════════════════════════════════
# TAB 1 — RECOMMENDER
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown(f"""
    <div style="margin:24px 0 16px;">
        <div style="font-size:13px;font-weight:700;color:{SUBTEXT};letter-spacing:2px;margin-bottom:8px;">
            CHOOSE YOUR MOVIE
        </div>
    </div>
    """, unsafe_allow_html=True)

    selected = st.selectbox("", movies["title"].values, label_visibility="collapsed")

    b1, b2, b3 = st.columns([1, 1, 3])
    with b1:
        find_clicked = st.button("🍿 Find Similar", use_container_width=True)
    with b2:
        details_clicked = st.button("ℹ️ View Details", use_container_width=True)

    if find_clicked:
        st.session_state.recommendations = recommend(selected)
        st.session_state.detail_title    = None
    if details_clicked:
        st.session_state.detail_title    = selected
        st.session_state.recommendations = []

    if st.session_state.detail_title and not st.session_state.recommendations:
        show_details(st.session_state.detail_title)

    if st.session_state.recommendations:
        names = st.session_state.recommendations
        st.markdown(f"""
        <div style="margin:28px 0 16px;">
            <div style="font-size:22px;font-weight:800;color:{TEXT};">🍿 Similar Movies</div>
            <div style="color:{SUBTEXT};font-size:13px;margin-top:4px;">
                Click <b>More Info</b> on any movie to see full details
            </div>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                mid_i = get_movie_id(names[i])
                poster_i = fetch_poster(names[i])
                st.image(poster_i, use_container_width=True)
                st.markdown(f"""
                <div style="background:linear-gradient(160deg,{CARD2},{CARD});
                            border:1px solid {BORDER};border-radius:0 0 16px 16px;
                            padding:12px 14px;text-align:center;
                            margin-top:-8px;margin-bottom:10px;">
                    <div style="font-size:12px;font-weight:700;color:{TEXT};line-height:1.4;">
                        {names[i]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("ℹ️ More Info", key=f"btn_{i}", use_container_width=True):
                    st.session_state.detail_title = names[i]
                    st.rerun()

        if st.session_state.detail_title:
            show_details(st.session_state.detail_title)

# ══════════════════════════════════════════════════════
# TAB 2 — TOP RATED BY GENRE
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown(f"""
    <div style="margin:24px 0 16px;">
        <div style="font-size:26px;font-weight:800;color:{TEXT};">🏆 Top Rated by Genre</div>
        <div style="color:{SUBTEXT};font-size:14px;margin-top:4px;">
            Select a genre to see the 10 highest rated movies
        </div>
    </div>
    """, unsafe_allow_html=True)

    selected_genre = st.selectbox("Pick a Genre:", all_genres,
                                  format_func=lambda g: f"{genre_emojis.get(g,'🎬')} {g}")

    if selected_genre:
        top_movies = get_top_by_genre(selected_genre)
        emoji = genre_emojis.get(selected_genre, "🎬")

        st.markdown(f"""
        <div style="margin:20px 0 16px;">
            <div style="font-size:20px;font-weight:800;color:{TEXT};">{emoji} Top 10 {selected_genre} Movies</div>
            <div style="color:{SUBTEXT};font-size:13px;">Minimum 100 votes required</div>
        </div>
        """, unsafe_allow_html=True)

        if top_movies.empty:
            st.warning(f"No movies found for: {selected_genre}")
        else:
            for idx, row in top_movies.iterrows():
                rank   = idx + 1
                rating = round(float(row["vote_average"]), 1)
                votes  = int(row["vote_count"])
                year   = str(row["release_date"])[:4] if str(row["release_date"]) != "nan" else "N/A"
                genres_str = " · ".join(get_genres(row["genres"])[:3])
                medal  = {1:"🥇",2:"🥈",3:"🥉"}.get(rank, f"#{rank}")
                pct    = int(rating * 10)
                stars  = "⭐" * int(rating/2) + "☆" * (5-int(rating/2))

                st.markdown(f"""
                <div style="background:linear-gradient(135deg,{CARD2},{CARD});
                            border:1px solid {BORDER};border-radius:16px;
                            padding:18px 24px;margin-bottom:10px;
                            box-shadow:0 4px 16px rgba(0,0,0,0.15);
                            display:flex;align-items:center;gap:20px;">
                    <div style="font-size:32px;min-width:44px;text-align:center;">{medal}</div>
                    <div style="flex:1;">
                        <div style="font-size:16px;font-weight:800;color:{TEXT};">{row['title']} <span style="color:{SUBTEXT};font-weight:400;font-size:14px;">({year})</span></div>
                        <div style="color:{SUBTEXT};font-size:12px;margin:4px 0;">{genres_str}</div>
                        <div style="font-size:16px;">{stars}</div>
                        <div style="background:{BORDER};border-radius:99px;height:4px;margin-top:8px;width:200px;">
                            <div style="background:linear-gradient(90deg,{ACCENT},{ACCENT2});width:{pct}%;height:4px;border-radius:99px;"></div>
                        </div>
                    </div>
                    <div style="text-align:right;min-width:80px;">
                        <div style="font-size:28px;font-weight:900;color:#f5c518;">{rating}</div>
                        <div style="color:{SUBTEXT};font-size:11px;">{votes:,} votes</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 3 — STATISTICS
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown(f"""
    <div style="margin:24px 0 20px;">
        <div style="font-size:26px;font-weight:800;color:{TEXT};">📊 Movie Statistics Dashboard</div>
        <div style="color:{SUBTEXT};font-size:14px;margin-top:4px;">Insights from 5,000 movies in the TMDB dataset</div>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def get_stats_data():
        df = movies_full.copy()
        genre_counts = {}
        for g_str in df["genres"]:
            for g in get_genres(g_str):
                genre_counts[g] = genre_counts.get(g, 0) + 1
        genre_df  = pd.DataFrame(list(genre_counts.items()), columns=["Genre","Count"]).sort_values("Count", ascending=False)
        budget_df = df[df["budget"]>1_000_000][["title","budget"]].sort_values("budget",ascending=False).head(10).copy()
        budget_df["budget_m"] = (budget_df["budget"]/1_000_000).round(1)
        dir_counts = {}
        for crew_str in df["crew"]:
            d = get_director(crew_str)
            if d != "N/A": dir_counts[d] = dir_counts.get(d,0)+1
        dir_df   = pd.DataFrame(list(dir_counts.items()),columns=["Director","Movies"]).sort_values("Movies",ascending=False).head(10)
        rating_df = df[df["vote_count"]>=50][["vote_average"]].copy()
        bvr_df   = df[(df["budget"]>1_000_000)&(df["revenue"]>1_000_000)][["title","budget","revenue","vote_average"]].copy()
        bvr_df["budget_m"]  = (bvr_df["budget"]/1_000_000).round(1)
        bvr_df["revenue_m"] = (bvr_df["revenue"]/1_000_000).round(1)
        return genre_df, budget_df, dir_df, rating_df, bvr_df

    genre_df, budget_df, dir_df, rating_df, bvr_df = get_stats_data()

    def apply_theme(fig, axes):
        fig.patch.set_facecolor(PLOT_PAP)
        for ax in (axes if isinstance(axes, list) else [axes]):
            ax.set_facecolor(PLOT_BG)
            ax.tick_params(colors=FONT_COL, labelsize=9)
            ax.xaxis.label.set_color(FONT_COL)
            ax.yaxis.label.set_color(FONT_COL)
            ax.title.set_color(FONT_COL)
            for spine in ax.spines.values(): spine.set_edgecolor(GRID_COL)
            ax.grid(color=GRID_COL, linestyle="--", linewidth=0.5, alpha=0.7)
        return fig

    # Summary metrics
    m1,m2,m3,m4,m5 = st.columns(5)
    with m1: st.metric("🎬 Total Movies",  f"{len(movies_full):,}")
    with m2: st.metric("🎭 Total Genres",  f"{len(genre_df)}")
    with m3: st.metric("⭐ Avg Rating",    f"{movies_full['vote_average'].mean():.1f}/10")
    with m4: st.metric("💰 Avg Budget",    f"${movies_full[movies_full['budget']>0]['budget'].mean()/1e6:.0f}M")
    with m5: st.metric("🏆 Highest Rated", f"{movies_full['vote_average'].max():.1f}/10")

    st.markdown("---")

    # Row 1
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div style='font-size:16px;font-weight:800;color:{TEXT};margin-bottom:12px;'>🎭 Most Common Genres</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,5))
        wedges, texts, autotexts = ax.pie(genre_df["Count"], labels=genre_df["Genre"],
            autopct="%1.1f%%", colors=plt.cm.Set3.colors, startangle=140,
            pctdistance=0.8, textprops={"fontsize":8,"color":FONT_COL})
        for at in autotexts: at.set_fontsize(7); at.set_color(FONT_COL)
        ax.set_facecolor(PLOT_BG); fig.patch.set_facecolor(PLOT_PAP)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with c2:
        st.markdown(f"<div style='font-size:16px;font-weight:800;color:{TEXT};margin-bottom:12px;'>💰 Top 10 Highest Budgeted Movies</div>", unsafe_allow_html=True)
        bd = budget_df.sort_values("budget_m")
        fig, ax = plt.subplots(figsize=(6,5))
        colors_list = [plt.cm.RdPu(x/bd["budget_m"].max()) for x in bd["budget_m"]]
        bars = ax.barh(bd["title"], bd["budget_m"], color=colors_list)
        ax.set_xlabel("Budget ($ Million)", color=FONT_COL)
        for bar, val in zip(bars, bd["budget_m"]):
            ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                    f"${val:.0f}M", va="center", fontsize=8, color=FONT_COL)
        apply_theme(fig, ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("---")

    # Row 2
    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"<div style='font-size:16px;font-weight:800;color:{TEXT};margin-bottom:12px;'>⭐ Rating Distribution</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,5))
        n, bins, patches = ax.hist(rating_df["vote_average"], bins=20, edgecolor="white", linewidth=0.5)
        for patch, height in zip(patches, n):
            patch.set_facecolor(plt.cm.plasma(height/max(n)*0.8+0.2))
        ax.set_xlabel("Rating", color=FONT_COL); ax.set_ylabel("Number of Movies", color=FONT_COL)
        apply_theme(fig, ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with c4:
        st.markdown(f"<div style='font-size:16px;font-weight:800;color:{TEXT};margin-bottom:12px;'>🎬 Top 10 Directors by Movies</div>", unsafe_allow_html=True)
        dd = dir_df.sort_values("Movies")
        fig, ax = plt.subplots(figsize=(6,5))
        colors_list = [plt.cm.cool(x/dd["Movies"].max()) for x in dd["Movies"]]
        bars = ax.barh(dd["Director"], dd["Movies"], color=colors_list)
        ax.set_xlabel("Number of Movies", color=FONT_COL)
        for bar, val in zip(bars, dd["Movies"]):
            ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                    str(val), va="center", fontsize=9, color=FONT_COL)
        apply_theme(fig, ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("---")

    # Row 3 — full width scatter
    st.markdown(f"<div style='font-size:16px;font-weight:800;color:{TEXT};margin-bottom:4px;'>💹 Budget vs Box Office Revenue</div>", unsafe_allow_html=True)
    st.caption("Each dot is a movie — size shows rating · red dashed line = break-even")
    fig, ax = plt.subplots(figsize=(12,5))
    sc = ax.scatter(bvr_df["budget_m"], bvr_df["revenue_m"],
                    c=bvr_df["vote_average"], s=bvr_df["vote_average"]*8,
                    cmap="plasma", alpha=0.6, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Rating", color=FONT_COL)
    cbar.ax.yaxis.set_tick_params(color=FONT_COL)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=FONT_COL)
    max_val = max(bvr_df["budget_m"].max(), bvr_df["revenue_m"].max())
    ax.plot([0,max_val],[0,max_val],"r--",linewidth=1.5,label="Break-even")
    ax.set_xlabel("Budget ($ Million)", color=FONT_COL)
    ax.set_ylabel("Revenue ($ Million)", color=FONT_COL)
    ax.legend(facecolor=PLOT_BG, labelcolor=FONT_COL, fontsize=9)
    apply_theme(fig, ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("📌 Movies **above** the red line made a profit. Movies **below** made a loss.")

# ══════════════════════════════════════════════════════
# TAB 4 — COMPARE MOVIES
# ══════════════════════════════════════════════════════
with tab4:
    st.markdown(f"""
    <div style="margin:24px 0 20px;">
        <div style="font-size:26px;font-weight:800;color:{TEXT};">⚖️ Movie Comparison</div>
        <div style="color:{SUBTEXT};font-size:14px;margin-top:4px;">Pick 2 movies and compare them head to head</div>
    </div>
    """, unsafe_allow_html=True)

    all_titles = sorted(movies["title"].tolist())
    c1, c2 = st.columns(2)
    with c1:
        movie1 = st.selectbox("🎬 First Movie",  all_titles, index=0, key="cmp1")
    with c2:
        movie2 = st.selectbox("🎬 Second Movie", all_titles, index=min(1,len(all_titles)-1), key="cmp2")

    compare_clicked = st.button("⚖️ Compare Now!", use_container_width=False)

    if compare_clicked:
        if movie1 == movie2:
            st.warning("Please select two different movies!")
        else:
            d1 = get_details(movie1)
            d2 = get_details(movie2)
            if not d1 or not d2:
                st.error("Could not find details for one of the movies.")
            else:
                # VS Header with posters
                p1 = fetch_poster(movie1)
                p2 = fetch_poster(movie2)

                pc1, pcvs, pc2 = st.columns([2, 1, 2])
                with pc1:
                    st.image(p1, use_container_width=True)
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,{CARD2},{CARD});
                                border:1px solid {BORDER};border-top:none;border-radius:0 0 16px 16px;
                                padding:16px;text-align:center;border-left:4px solid {ACCENT};">
                        <div style="font-size:16px;font-weight:900;color:{TEXT};">{d1['title']}</div>
                        <div style="color:{SUBTEXT};font-size:12px;margin-top:4px;">📅 {d1['year']} · 🌐 {d1['language']}</div>
                    </div>""", unsafe_allow_html=True)
                with pcvs:
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;justify-content:center;height:100%;">
                        <div style="background:linear-gradient(135deg,{ACCENT},{ACCENT2});
                                    border-radius:50%;width:64px;height:64px;
                                    display:flex;align-items:center;justify-content:center;
                                    font-size:20px;font-weight:900;color:white;
                                    box-shadow:0 0 30px rgba(233,69,96,0.5);">VS</div>
                    </div>""", unsafe_allow_html=True)
                with pc2:
                    st.image(p2, use_container_width=True)
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,{CARD2},{CARD});
                                border:1px solid {BORDER};border-top:none;border-radius:0 0 16px 16px;
                                padding:16px;text-align:center;border-left:4px solid {ACCENT2};">
                        <div style="font-size:16px;font-weight:900;color:{TEXT};">{d2['title']}</div>
                        <div style="color:{SUBTEXT};font-size:12px;margin-top:4px;">📅 {d2['year']} · 🌐 {d2['language']}</div>
                    </div>""", unsafe_allow_html=True)

                def winner(v1, v2, higher=True):
                    if v1==v2: return "🤝","🤝"
                    return ("🏆","  ") if (v1>v2)==higher else ("  ","🏆")

                def cmp_row(label, val1, val2, w1, w2, color1=ACCENT, color2=ACCENT2):
                    st.markdown(f"""
                    <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:12px;
                                align-items:center;margin:10px 0;">
                        <div style="background:{CARD};border:1px solid {BORDER};
                                    border-radius:14px;padding:16px 20px;text-align:center;
                                    {'border:2px solid '+color1+';box-shadow:0 0 15px rgba(233,69,96,0.2);' if w1=='🏆' else ''}">
                            <div style="font-size:11px;color:{SUBTEXT};letter-spacing:1px;">{w1}</div>
                            <div style="font-size:20px;font-weight:800;color:{TEXT};margin:4px 0;">{val1}</div>
                        </div>
                        <div style="text-align:center;font-size:13px;font-weight:700;
                                    color:{SUBTEXT};padding:0 8px;min-width:60px;">{label}</div>
                        <div style="background:{CARD};border:1px solid {BORDER};
                                    border-radius:14px;padding:16px 20px;text-align:center;
                                    {'border:2px solid '+color2+';box-shadow:0 0 15px rgba(123,97,255,0.2);' if w2=='🏆' else ''}">
                            <div style="font-size:11px;color:{SUBTEXT};letter-spacing:1px;">{w2}</div>
                            <div style="font-size:20px;font-weight:800;color:{TEXT};margin:4px 0;">{val2}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Rating
                st.markdown(f"<div style='font-size:16px;font-weight:800;color:{TEXT};margin:20px 0 8px;'>⭐ Rating</div>", unsafe_allow_html=True)
                w1,w2 = winner(d1["rating"], d2["rating"])
                cmp_row("RATING", f"{d1['rating']}/10", f"{d2['rating']}/10", w1, w2)
                rc1,rc2 = st.columns(2)
                with rc1: st.progress(int(d1["rating"]*10))
                with rc2: st.progress(int(d2["rating"]*10))

                # Runtime
                st.markdown(f"<div style='font-size:16px;font-weight:800;color:{TEXT};margin:16px 0 8px;'>⏱ Runtime</div>", unsafe_allow_html=True)
                rt1 = d1["runtime"] if d1["runtime"] else 0
                rt2 = d2["runtime"] if d2["runtime"] else 0
                w1,w2 = winner(rt1, rt2, higher=False)
                cmp_row("RUNTIME", f"{rt1} min" if rt1 else "N/A", f"{rt2} min" if rt2 else "N/A", w1, w2)

                # Budget
                st.markdown(f"<div style='font-size:16px;font-weight:800;color:{TEXT};margin:16px 0 8px;'>💰 Budget</div>", unsafe_allow_html=True)
                w1,w2 = winner(d1["budget"], d2["budget"])
                cmp_row("BUDGET",
                        f"${d1['budget']:,}" if d1["budget"]>0 else "N/A",
                        f"${d2['budget']:,}" if d2["budget"]>0 else "N/A", w1, w2)

                # Revenue
                st.markdown(f"<div style='font-size:16px;font-weight:800;color:{TEXT};margin:16px 0 8px;'>🏦 Box Office</div>", unsafe_allow_html=True)
                w1,w2 = winner(d1["revenue"], d2["revenue"])
                cmp_row("REVENUE",
                        f"${d1['revenue']:,}" if d1["revenue"]>0 else "N/A",
                        f"${d2['revenue']:,}" if d2["revenue"]>0 else "N/A", w1, w2)

                # Genres
                st.markdown(f"<div style='font-size:16px;font-weight:800;color:{TEXT};margin:20px 0 12px;'>🎭 Genres</div>", unsafe_allow_html=True)
                g1s = set(d1["genres"]); g2s = set(d2["genres"]); shared = g1s & g2s
                gc1, gcm, gc2 = st.columns([2,1,2])
                with gc1:
                    for g in d1["genres"]:
                        ic = "✅" if g in shared else "🔵"
                        st.markdown(f"{ic} `{g}`")
                with gcm:
                    st.markdown(f"**🤝 Shared**" if shared else "**No shared**")
                    for g in shared: st.markdown(f"`{g}`")
                with gc2:
                    for g in d2["genres"]:
                        ic = "✅" if g in shared else "🔴"
                        st.markdown(f"{ic} `{g}`")

                # Cast
                st.markdown(f"<div style='font-size:16px;font-weight:800;color:{TEXT};margin:20px 0 12px;'>🌟 Cast</div>", unsafe_allow_html=True)
                cast1s = set(d1["cast"]); cast2s = set(d2["cast"]); common = cast1s & cast2s
                cc1, ccm, cc2 = st.columns([2,1,2])
                with cc1:
                    for c in d1["cast"]:
                        ic = "✅" if c in common else "👤"
                        st.markdown(f"{ic} {c}")
                with ccm:
                    if common:
                        st.markdown("**🌟 Both**")
                        for c in common: st.markdown(f"✅ {c}")
                    else: st.caption("No common cast")
                with cc2:
                    for c in d2["cast"]:
                        ic = "✅" if c in common else "👤"
                        st.markdown(f"{ic} {c}")

                # Verdict
                st.markdown(f"<div style='font-size:20px;font-weight:800;color:{TEXT};margin:24px 0 16px;'>🏆 Overall Verdict</div>", unsafe_allow_html=True)
                s1=s2=0
                if d1["rating"]  > d2["rating"]:  s1+=1
                elif d2["rating"]  > d1["rating"]: s2+=1
                if d1["revenue"] > d2["revenue"]: s1+=1
                elif d2["revenue"] > d1["revenue"]: s2+=1
                if 0<d1["runtime"]<d2["runtime"]:  s1+=1
                elif 0<d2["runtime"]<d1["runtime"]: s2+=1
                if len(g1s-g2s)>len(g2s-g1s): s1+=1
                elif len(g2s-g1s)>len(g1s-g2s): s2+=1

                vc1, vc2 = st.columns(2)
                with vc1:
                    color = ACCENT if s1>=s2 else BORDER
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,{CARD2},{CARD});
                                border:2px solid {color};border-radius:18px;padding:28px;text-align:center;
                                box-shadow:{'0 0 30px rgba(233,69,96,0.3)' if s1>s2 else 'none'};">
                        <div style="font-size:48px;">{'🏆' if s1>s2 else '🥈' if s1==s2 else '❌'}</div>
                        <div style="font-size:18px;font-weight:800;color:{TEXT};margin:8px 0;">{d1['title']}</div>
                        <div style="font-size:28px;font-weight:900;color:{ACCENT};">{s1}/4</div>
                    </div>
                    """, unsafe_allow_html=True)
                with vc2:
                    color = ACCENT2 if s2>=s1 else BORDER
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,{CARD2},{CARD});
                                border:2px solid {color};border-radius:18px;padding:28px;text-align:center;
                                box-shadow:{'0 0 30px rgba(123,97,255,0.3)' if s2>s1 else 'none'};">
                        <div style="font-size:48px;">{'🏆' if s2>s1 else '🥈' if s1==s2 else '❌'}</div>
                        <div style="font-size:18px;font-weight:800;color:{TEXT};margin:8px 0;">{d2['title']}</div>
                        <div style="font-size:28px;font-weight:900;color:{ACCENT2};">{s2}/4</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                if s1==s2:   st.info("🤝 It's a tie! Both movies are equally matched.")
                elif s1>s2:  st.success(f"🏆 **{d1['title']}** wins the comparison!")
                else:        st.success(f"🏆 **{d2['title']}** wins the comparison!")