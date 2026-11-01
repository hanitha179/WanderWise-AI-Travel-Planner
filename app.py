from __future__ import annotations

from pathlib import Path
import math
import requests
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "indonesia_destinations.csv"

st.set_page_config(page_title="WanderWise | Indonesia Trip Planner", page_icon="🧭", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@500;600;700&display=swap');
:root{--forest:#173f35;--sage:#dce9df;--mint:#eef5ef;--cream:#faf9f5;--ink:#20352e;--muted:#718078;--gold:#d7a75b}
.stApp{background:var(--cream);color:var(--ink)}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
h1,h2,h3{font-family:'Playfair Display',serif!important;color:var(--forest)!important}
.block-container{max-width:1320px;padding-top:1.2rem;padding-bottom:4rem}
[data-testid="stSidebar"]{background:#edf3ec;border-right:1px solid #dce7dc}
[data-testid="stSidebar"] h2{font-family:'Playfair Display',serif!important}
.stButton>button,.stDownloadButton>button{border-radius:12px;border:0;background:var(--forest);color:white;font-weight:700;padding:.65rem 1rem;min-height:2.8rem}
.stButton>button:hover,.stDownloadButton>button:hover{background:#285b4d;color:white;border:0}
[data-testid="stMetric"]{background:white;border:1px solid #e6ebe5;padding:16px 18px;border-radius:16px;box-shadow:0 4px 18px #183f3510}
[data-testid="stMetricLabel"]{color:var(--muted)}
[data-testid="stDataFrame"]{border:1px solid #e5ebe4;border-radius:14px;overflow:hidden}
.hero{background:linear-gradient(115deg,#153f35 0%,#2f6854 60%,#709477 100%);padding:2.5rem 2.6rem;border-radius:26px;color:white;margin:.5rem 0 1.6rem;position:relative;overflow:hidden}
.hero h1{color:white!important;font-size:3.6rem!important;margin:.25rem 0!important;line-height:1.05}
.hero p{color:#e2eee5;font-size:1.05rem;max-width:650px;margin:.6rem 0 0}
.eyebrow{letter-spacing:.18em;text-transform:uppercase;color:#c7dfca;font-size:.72rem;font-weight:800}
.hero-tag{display:inline-block;margin-top:1.2rem;padding:.45rem .8rem;border:1px solid #ffffff55;border-radius:999px;color:#f4f7ef;font-size:.78rem}
.section-kicker{font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:#64836e;font-weight:800;margin-bottom:.2rem}
.place-card{background:white;border:1px solid #e5ebe4;border-radius:16px;padding:1rem 1.1rem;height:100%;box-shadow:0 5px 20px #193d3010}
.place-card .place-type{font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;color:#718c76;font-weight:800}
.place-card .place-name{font-family:'Playfair Display',serif;font-size:1.2rem;color:#173f35;margin:.35rem 0}
.place-card .place-meta{color:#718078;font-size:.85rem}
.info-strip{background:#edf4ed;border:1px solid #dce8dc;border-radius:14px;padding:1rem 1.2rem;color:#355344}
hr{border-color:#e5ebe4!important}
</style>
""", unsafe_allow_html=True)
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    required = {"city", "name", "category", "description", "rating", "estimated_cost_idr", "lat", "lon", "visit_minutes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    return df.dropna(subset=["city", "name", "lat", "lon"]).copy()

INTEREST_TERMS = {
    "Beaches": "beach coastline island sea swimming snorkeling sand sunset",
    "Adventure": "adventure hiking trekking surfing diving cycling outdoor",
    "Culture": "culture tradition local village crafts performance heritage",
    "Nature": "nature forest waterfall volcano mountain wildlife garden scenery",
    "History": "history temple palace monument museum archaeology historic",
    "Food": "food culinary market street food cooking local cuisine",
    "Photography": "viewpoint panorama landscape sunrise sunset photography",
    "Family": "family children park aquarium recreation activities",
}


def recommend_destinations(df: pd.DataFrame, interests: list[str], budget_idr: int, days: int, min_rating: float) -> pd.DataFrame:
    """Content-based ranking: TF-IDF/cosine relevance + rating + affordability."""
    if df.empty:
        return df.copy()
    d = df.copy()
    corpus = (d["name"].fillna("").astype(str) + " " + d["category"].fillna("").astype(str) + " " + d["description"].fillna("").astype(str)).tolist()
    query = " ".join(INTEREST_TERMS.get(x, x) for x in interests) or "popular travel attractions and places to visit"
    try:
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(corpus + [query])
        relevance = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    except ValueError:
        relevance = [0.0] * len(d)
    d["relevance"] = relevance
    d["rating_score"] = (pd.to_numeric(d["rating"], errors="coerce").fillna(0) / 5).clip(0, 1)
    # Allocate a planning allowance per attraction from the user's overall trip budget.
    allowance_per_stop = max(float(budget_idr) / max(days * 3, 1), 1.0)
    d["affordability"] = (1 - pd.to_numeric(d["estimated_cost_idr"], errors="coerce").fillna(allowance_per_stop) / allowance_per_stop).clip(0, 1)
    d["score"] = 0.65 * d["relevance"] + 0.25 * d["rating_score"] + 0.10 * d["affordability"]
    d = d[d["rating"] >= min_rating]
    return d.sort_values(["score", "rating"], ascending=False).reset_index(drop=True)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def generate_itinerary(recommendations: pd.DataFrame, days: int, travellers: int, budget_idr: int) -> pd.DataFrame:
    """Automatically allocate high-ranked attractions and order each day by proximity."""
    if recommendations.empty:
        return pd.DataFrame()
    chosen = recommendations.head(min(len(recommendations), days * 3)).copy()
    remaining = chosen.to_dict("records")
    rows = []
    for day in range(1, days + 1):
        if not remaining:
            break
        # Balanced distribution: approximately three attractions per day.
        days_left = days - day + 1
        slots = min(3, math.ceil(len(remaining) / days_left))
        current = (float(remaining[0]["lat"]), float(remaining[0]["lon"]))
        day_items = []
        for _ in range(min(slots, len(remaining))):
            nxt = min(remaining, key=lambda x: haversine_km(current[0], current[1], float(x["lat"]), float(x["lon"])))
            remaining.remove(nxt)
            day_items.append(nxt)
            current = (float(nxt["lat"]), float(nxt["lon"]))
        start_minutes = 9 * 60
        for index, item in enumerate(day_items):
            travel_minutes = 0 if index == 0 else 30  # indicative transfer allowance, not live routing
            start_minutes += travel_minutes
            hour, minute = divmod(start_minutes, 60)
            rows.append({
                "Day": f"Day {day}", "Time": f"{hour:02d}:{minute:02d}",
                "Destination": item["name"], "Category": item["category"],
                "Visit duration (min)": int(item["visit_minutes"]),
                "Estimated entry cost (IDR)": int(item["estimated_cost_idr"]) * travellers,
                "Rating": float(item["rating"]), "Latitude": item["lat"], "Longitude": item["lon"],
            })
            start_minutes += int(item["visit_minutes"])
    plan = pd.DataFrame(rows)
    if not plan.empty:
        plan.attrs["estimated_entry_total"] = int(plan["Estimated entry cost (IDR)"].sum())
        plan.attrs["budget_idr"] = int(budget_idr)
    return plan


OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def search_nearby(lat: float, lon: float, kind: str, radius_m: int = 3000) -> pd.DataFrame:
    if kind == "Hotels":
        selector = 'nwr(around:{r},{lat},{lon})["tourism"~"^(hotel|hostel|guest_house|motel|apartment|resort)$"]["name"];'
    else:
        selector = 'nwr(around:{r},{lat},{lon})["amenity"~"^(restaurant|cafe|fast_food|food_court)$"]["name"];'
    query = "[out:json][timeout:25];(" + selector.format(r=int(radius_m), lat=lat, lon=lon) + ");out center tags 150;"
    last_error = None
    for url in OVERPASS_URLS:
        try:
            response = requests.post(url, data={"data": query}, headers={"User-Agent": "WanderWiseStudentProject/1.0"}, timeout=35)
            response.raise_for_status()
            elements = response.json().get("elements", [])
            rows = []
            for el in elements:
                tags = el.get("tags", {})
                center = el.get("center", {})
                plat = el.get("lat", center.get("lat"))
                plon = el.get("lon", center.get("lon"))
                if plat is None or plon is None:
                    continue
                rows.append({
                    "Name": tags.get("name", "Unnamed place"),
                    "Type": tags.get("tourism", tags.get("amenity", "")),
                    "Distance (km)": round(haversine_km(lat, lon, float(plat), float(plon)), 2),
                    "Website": tags.get("website", ""),
                    "Map": f"https://www.openstreetmap.org/?mlat={plat}&mlon={plon}#map=17/{plat}/{plon}",
                    "Latitude": plat, "Longitude": plon,
                })
            return pd.DataFrame(rows).sort_values("Distance (km)") if rows else pd.DataFrame()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"OpenStreetMap is temporarily unavailable. Please retry later. ({last_error})")


try:
    data = load_data()
except Exception as exc:
    st.error(f"Could not load the destination dataset: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("## 🧭 WanderWise")
    st.caption("YOUR PERSONAL INDONESIA TRIP DESIGNER")
    st.divider()
    st.markdown("### Build your trip")
    cities = sorted(data["city"].dropna().unique().tolist())
    city = st.selectbox("Destination city", cities)
    interests = st.multiselect("What do you love?", list(INTEREST_TERMS), default=["Culture", "Nature"])
    budget = st.number_input("Total trip budget (IDR)", min_value=100000, max_value=100000000, value=5000000, step=100000)
    days = st.slider("Trip duration", 1, 10, 3, format="%d days")
    travellers = st.number_input("Travellers", min_value=1, max_value=20, value=2)
    min_rating = st.slider("Minimum destination rating", 1.0, 5.0, 3.5, 0.5)
    st.divider()
    st.caption("Starter dataset values are illustrative, not live prices or verified review scores.")

city_data = data[data["city"] == city].copy()
recommendations = recommend_destinations(city_data, interests, int(budget), int(days), float(min_rating))

st.markdown('''<div class="hero">
<div class="eyebrow">A LITTLE MORE DISCOVERY. A LOT LESS PLANNING.</div>
<h1>WanderWise</h1>
<p>Your next Indonesian escape, thoughtfully planned. Discover places that match your interests, shape a day-by-day itinerary and explore nearby stays and food spots.</p>
<span class="hero-tag">✦ AI-assisted destination discovery &nbsp; · &nbsp; Made for curious travellers</span>
</div>''', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Destination", city)
m2.metric("Trip length", f"{days} days")
m3.metric("Travellers", int(travellers))
m4.metric("Total budget", f"Rp {budget:,.0f}")

st.markdown('<div class="section-kicker">01 / DISCOVER</div>', unsafe_allow_html=True)
st.header("Places picked around you")
st.caption("Recommendations are ranked using your interests, destination ratings and an affordability heuristic.")
st.markdown('<div class="info-strip">✦ <b>How the AI works:</b> TF-IDF and cosine similarity compare your interests with each destination description. The final score blends interest match (65%), illustrative rating (25%) and affordability (10%).</div>', unsafe_allow_html=True)
st.write("")

if recommendations.empty:
    st.warning("No destinations match this rating threshold. Try lowering the minimum rating or choosing another city.")
else:
    top = recommendations.head(3)
    cols = st.columns(3)
    for col, (_, place) in zip(cols, top.iterrows()):
        with col:
            st.markdown(f'''<div class="place-card"><div class="place-type">{place['category']} · {place['city']}</div><div class="place-name">{place['name']}</div><div class="place-meta">★ {float(place['rating']):.1f}/5 &nbsp; · &nbsp; Approx. Rp {int(place['estimated_cost_idr']):,} / person</div></div>''', unsafe_allow_html=True)
    with st.expander("View all ranked recommendations", expanded=True):
        display = recommendations[["name", "category", "rating", "estimated_cost_idr", "relevance", "score"]].head(12).copy()
        display.columns = ["Destination", "Category", "Rating (demo)", "Estimated cost / person (IDR)", "Interest match", "Rank score"]
        display["Interest match"] = (display["Interest match"] * 100).round(1).astype(str) + "%"
        display["Rank score"] = display["Rank score"].round(3)
        st.dataframe(display, hide_index=True, use_container_width=True)
    best = recommendations.iloc[0]
    st.success(f"Best match: {best['name']} — {best['city']} ({best['rating']:.1f}/5 illustrative rating)")

st.divider()
st.markdown('<div class="section-kicker">02 / BUILD YOUR DAYS</div>', unsafe_allow_html=True)
st.header("Your trip, mapped into moments")
st.write("Create a practical starting itinerary from your highest-ranked destinations. Stops are grouped by day and ordered using approximate geographic proximity.")
if st.button("✦ Generate my itinerary", type="primary", disabled=recommendations.empty):
    st.session_state["generated_plan"] = generate_itinerary(recommendations, int(days), int(travellers), int(budget))
    st.session_state["plan_city"] = city

plan = st.session_state.get("generated_plan")
if isinstance(plan, pd.DataFrame) and not plan.empty and st.session_state.get("plan_city") == city:
    entry_total = int(plan["Estimated entry cost (IDR)"].sum())
    p1, p2, p3 = st.columns(3)
    p1.metric("Planned stops", len(plan))
    p2.metric("Estimated entry costs", f"Rp {entry_total:,.0f}")
    p3.metric("Budget after entry costs", f"Rp {max(int(budget)-entry_total, 0):,.0f}")
    for day_name, group in plan.groupby("Day", sort=False):
        with st.expander(f"📍 {day_name}  ·  {len(group)} stops", expanded=True):
            st.dataframe(group.drop(columns=["Day", "Latitude", "Longitude"]), hide_index=True, use_container_width=True)
    st.download_button("⬇ Download itinerary CSV", plan.to_csv(index=False).encode("utf-8"), f"wanderwise_{city.lower().replace(' ', '_')}_itinerary.csv", "text/csv")
    st.caption("The schedule uses estimated visit durations and an indicative 30-minute transfer allowance. It is not live route optimisation. Food, lodging, transport, flights and opening hours are not included in the entry-cost subtotal.")
else:
    st.info("Your day-wise itinerary will appear here after you generate it.")

st.divider()
st.markdown('<div class="section-kicker">03 / EXPLORE NEARBY</div>', unsafe_allow_html=True)
st.header("Find a place to stay or eat")
st.write("Search OpenStreetMap community listings around one of your recommended attractions.")
if recommendations.empty:
    st.info("Choose a city and rating threshold that returns destinations to search nearby places.")
else:
    place_options = recommendations.head(12)["name"].tolist()
    selected_name = st.selectbox("Search around this attraction", place_options)
    selected = recommendations[recommendations["name"] == selected_name].iloc[0]
    nearby_kind = st.radio("Looking for", ["Hotels", "Restaurants"], horizontal=True)
    radius = st.select_slider("Search distance", options=[500, 1000, 2000, 3000, 5000, 10000], value=3000, format_func=lambda x: f"{x:,} m")
    if st.button("Search nearby places"):
        with st.spinner(f"Looking for {nearby_kind.lower()} near {selected_name}..."):
            try:
                found = search_nearby(float(selected["lat"]), float(selected["lon"]), nearby_kind, int(radius))
                st.session_state["nearby_results"] = found
                st.session_state["nearby_query"] = (selected_name, nearby_kind)
            except RuntimeError as exc:
                st.warning(str(exc))
    found = st.session_state.get("nearby_results")
    if isinstance(found, pd.DataFrame):
        if found.empty:
            st.info("No mapped places were found in this radius. Try a larger radius or another attraction.")
        else:
            st.success(f"Found {len(found)} mapped places near {selected_name}.")
            st.dataframe(found.drop(columns=["Latitude", "Longitude"]), hide_index=True, use_container_width=True, column_config={"Map": st.column_config.LinkColumn("Map", display_text="Open map"), "Website": st.column_config.LinkColumn("Website")})
            st.map(found[["Latitude", "Longitude"]].rename(columns={"Latitude": "lat", "Longitude": "lon"}))
            st.caption("Listings come from OpenStreetMap community mapping. They do not confirm current opening hours, room availability, prices or ratings.")

st.divider()
st.markdown("<div style='text-align:center;color:#849188;font-size:.82rem;padding:1rem 0'>WanderWise · Explore thoughtfully, travel freely · Student project prototype</div>", unsafe_allow_html=True)
