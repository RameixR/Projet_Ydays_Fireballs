import streamlit as st
import json
import os
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
API_URL   = "https://ssd-api.jpl.nasa.gov/fireball.api"
DATA_PATH = "data/fireballs.json"

ZONES = {
    "Amérique du Nord":   ( 15,  90, -170,  -50),
    "Amérique du Sud":    (-60,  15,  -82,  -34),
    "Europe":             ( 35,  72,  -25,   45),
    "Afrique":            (-40,  37,  -18,   52),
    "Moyen-Orient":       ( 12,  42,   35,   65),
    "Asie Centrale":      ( 30,  60,   50,  100),
    "Asie de l'Est":      ( 15,  55,  100,  145),
    "Asie du Sud-Est":    (-10,  25,   95,  145),
    "Océanie":            (-50,  -5,  110,  180),
    "Arctique":           ( 60,  90, -180,  180),
    "Antarctique":        (-90, -60, -180,  180),
    "Océan Pacifique":    (-60,  60, -180,  -82),
    "Océan Atlantique":   (-60,  60,  -50,  -18),
    "Océan Indien":       (-60,  25,   52,   95),
}

st.set_page_config(
    page_title="☄️ Fireballs NASA",
    page_icon="☄️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE DB.py
# ─────────────────────────────────────────────────────────────────────────────
def CréationDB():
    os.makedirs("data", exist_ok=True)
    r = requests.get(API_URL, timeout=30)
    r.raise_for_status()
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(r.json(), f, indent=4, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE DBclean.py
# ─────────────────────────────────────────────────────────────────────────────
def Nettoyer_donnees(data: dict) -> tuple[dict, int]:
    original = len(data["data"])
    seen, cleaned = set(), []
    for entry in data["data"]:
        if all(x is None or x == "" for x in entry):
            continue
        key = tuple(str(x) for x in entry)
        if key not in seen:
            seen.add(key)
            cleaned.append(entry)
    try:
        cleaned.sort(key=lambda x: datetime.strptime(x[0], "%Y-%m-%d %H:%M:%S"))
    except Exception:
        pass
    data["data"]  = cleaned
    data["count"] = str(len(cleaned))
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return data, original - len(cleaned)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE DBsupp.py
# ─────────────────────────────────────────────────────────────────────────────
def SuppDB():
    if os.path.exists(DATA_PATH):
        os.remove(DATA_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE DBtri.py
# ─────────────────────────────────────────────────────────────────────────────
def Tri_moyenne(data: dict, date_min: datetime, date_max: datetime) -> list:
    result = [
        e for e in data["data"]
        if date_min <= datetime.strptime(e[0], "%Y-%m-%d %H:%M:%S") <= date_max
    ]
    result.sort(key=lambda x: datetime.strptime(x[0], "%Y-%m-%d %H:%M:%S"))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# REGROUPEMENT PAR ZONE
# ─────────────────────────────────────────────────────────────────────────────
def get_zone(lat: float, lon: float) -> str:
    for nom, (lat_min, lat_max, lon_min, lon_max) in ZONES.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return nom
    return "Autre"


def regrouper_par_zone(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.dropna(subset=["lat", "lon", "lat-dir", "lon-dir"]).copy()
    df["lat_dec"] = df.apply(
        lambda r: r["lat"] * (-1 if r["lat-dir"] == "S" else 1), axis=1)
    df["lon_dec"] = df.apply(
        lambda r: r["lon"] * (-1 if r["lon-dir"] == "W" else 1), axis=1)
    df["zone"] = df.apply(lambda r: get_zone(r["lat_dec"], r["lon_dec"]), axis=1)

    agg = (
        df.groupby("zone")
        .agg(
            nb_impacts      =("energy", "count"),
            energie_totale  =("energy", "sum"),
            energie_moyenne =("energy", "mean"),
            energie_max     =("energy", "max"),
        )
        .reset_index()
        .sort_values("nb_impacts", ascending=False)
        .reset_index(drop=True)
    )
    agg.index += 1
    return agg, df


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────
def load_raw() -> dict | None:
    if not os.path.exists(DATA_PATH):
        return None
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def to_dataframe(data: dict) -> pd.DataFrame:
    df = pd.DataFrame(data["data"], columns=data["fields"])
    df["date"] = pd.to_datetime(df["date"])
    for col in ["energy", "impact-e", "lat", "lon", "alt", "vel"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("☄️ NASA Fireballs")
st.sidebar.markdown("---")
st.sidebar.subheader("Base de données")

if st.sidebar.button("📥 Télécharger les données", use_container_width=True):
    with st.spinner("Connexion à l'API NASA…"):
        try:
            CréationDB()
            st.sidebar.success("Données téléchargées !")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Erreur : {e}")

raw = load_raw()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────
st.title("☄️ Où tombent le plus les météorites ?")
st.caption("Source : NASA/JPL Fireball Data API")

if raw is None:
    st.info("Aucune donnée. Cliquez sur **Télécharger les données** dans la barre latérale.")
    st.stop()

df_all = to_dataframe(raw)

# ── FILTRE PAR DATE ───────────────────────────────────────────────────────────
st.subheader("🗓️ Période analysée")

date_min_db = df_all["date"].min().date()
date_max_db = df_all["date"].max().date()

c1, c2 = st.columns(2)
d_min = c1.date_input(
    "Date de début",
    value=date_min_db,
    min_value=date_min_db,
    max_value=date_max_db,
    key="date_min",
    format="DD/MM/YYYY",
)
d_max = c2.date_input(
    "Date de fin",
    value=date_max_db,
    min_value=date_min_db,
    max_value=date_max_db,
    key="date_max",
    format="DD/MM/YYYY",
)

date_min_filtre = datetime(d_min.year, d_min.month, d_min.day, 0, 0, 0)
date_max_filtre = datetime(d_max.year, d_max.month, d_max.day, 23, 59, 59)

entries = Tri_moyenne(raw, date_min_filtre, date_max_filtre)
df = to_dataframe({"fields": raw["fields"], "data": entries})

st.caption(
    f"**{d_min.strftime('%d/%m/%Y')} → {d_max.strftime('%d/%m/%Y')}** — "
    f"**{len(df):,}** événements — "
    f"**{df['lat'].notna().sum():,}** avec coordonnées géographiques"
)
st.divider()

# ── REGROUPEMENT ──────────────────────────────────────────────────────────────
agg, df_geo = regrouper_par_zone(df)
zone_top = agg.iloc[0]["zone"]
nb_top   = int(agg.iloc[0]["nb_impacts"])

# Bandeau résultat principal
st.markdown(
    f"""
    <div style="background:#1a1a2e;border-left:6px solid #e94560;
                padding:18px 24px;border-radius:8px;margin-bottom:8px;">
        <span style="font-size:1rem;color:#aaa;">Zone avec le plus d'impacts détectés :</span><br>
        <span style="font-size:2rem;font-weight:700;color:#e94560;">🏆 {zone_top}</span>
        <span style="font-size:1.3rem;color:#fff;margin-left:16px;">{nb_top} impacts</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── CLASSEMENT + GRAPHE ───────────────────────────────────────────────────────
st.subheader("📊 Classement complet des zones")

col_table, col_bar = st.columns([1, 1.5])

with col_table:
    # Renommage des colonnes pour l'affichage
    agg_display = agg.rename(columns={
        "zone":             "Zone",
        "nb_impacts":       "Impacts",
        "energie_totale":   "Énergie totale (GJ)",
        "energie_moyenne":  "Énergie moy. (GJ)",
        "energie_max":      "Énergie max (GJ)",
    })
    # Arrondi des colonnes numériques
    agg_display["Énergie totale (GJ)"] = agg_display["Énergie totale (GJ)"].round(0).astype(int)
    agg_display["Énergie moy. (GJ)"]   = agg_display["Énergie moy. (GJ)"].round(1)
    agg_display["Énergie max (GJ)"]    = agg_display["Énergie max (GJ)"].round(0).astype(int)

    st.dataframe(agg_display, width=600, height=520)

with col_bar:
    fig_bar = px.bar(
        agg.sort_values("nb_impacts"),
        x="nb_impacts",
        y="zone",
        orientation="h",
        color="nb_impacts",
        color_continuous_scale="YlOrRd",
        text="nb_impacts",
        labels={"nb_impacts": "Nombre d'impacts", "zone": ""},
        title="Impacts par zone géographique",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        height=520,
        xaxis=dict(showgrid=True, gridcolor="#333"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── CARTE ─────────────────────────────────────────────────────────────────────
st.subheader("🗺️ Carte mondiale des impacts")

fig_map = px.scatter_geo(
    df_geo,
    lat="lat_dec",
    lon="lon_dec",
    color="zone",
    size=df_geo["energy"].clip(upper=300).fillna(5),
    hover_name="zone",
    hover_data={"energy": True, "lat_dec": False, "lon_dec": False},
    projection="natural earth",
    title="Chaque point = un impact (taille proportionnelle à l'énergie)",
)
fig_map.update_layout(height=560, legend_title="Zone")
st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# ── ZOOM SUR UNE ZONE ─────────────────────────────────────────────────────────
st.subheader("🔍 Détail d'une zone")

zone_choisie = st.selectbox(
    "Choisir une zone à explorer",
    options=agg["zone"].tolist(),
    format_func=lambda z: f"{'🥇' if z == zone_top else '📍'} {z}",
)

df_zone = df_geo[df_geo["zone"] == zone_choisie].copy()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Impacts",           f"{len(df_zone):,}")
m2.metric("Énergie totale",    f"{df_zone['energy'].sum():,.0f} GJ")
m3.metric("Énergie moyenne",   f"{df_zone['energy'].mean():.1f} GJ")
m4.metric("Énergie maximale",  f"{df_zone['energy'].max():.0f} GJ")

# Évolution par année
par_an = (
    df_zone.groupby(df_zone["date"].dt.year)
    .size()
    .reset_index(name="impacts")
    .rename(columns={"date": "annee"})
)
fig_an = px.bar(
    par_an, x="annee", y="impacts",
    labels={"annee": "Année", "impacts": "Impacts"},
    title=f"Impacts par année — {zone_choisie}",
    color_discrete_sequence=["#e94560"],
)
fig_an.update_layout(plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_an, use_container_width=True)

# Liste des événements
with st.expander(f"📋 Voir les {len(df_zone)} événements de cette zone"):
    st.dataframe(
        df_zone[["date", "energy", "impact-e", "lat_dec", "lon_dec", "vel", "alt"]]
        .rename(columns={"lat_dec": "Latitude", "lon_dec": "Longitude"})
        .sort_values("energy", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
    )