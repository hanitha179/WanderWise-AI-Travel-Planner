# WanderWise — AI Indonesia Travel Planner

A polished Streamlit travel-planning website that recommends Indonesian destinations, creates day-wise itineraries, estimates attraction entry costs, and searches OpenStreetMap for nearby hotels and restaurants.

## Run locally on macOS (including Apple Silicon)

1. Extract the project ZIP.
2. Open Terminal and move into the project folder:

   ```bash
   cd path/to/WanderWise_AI_Travel_Planner
   ```

3. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. Start the website:

   ```bash
   streamlit run app.py
   ```

6. Open the local address printed in Terminal (normally http://localhost:8501).

## Website features

- **Personalised discovery:** choose a city and interests; recommendations use TF-IDF and cosine similarity.
- **Ranking:** blends interest relevance (65%), illustrative rating (25%), and affordability (10%).
- **Itinerary builder:** allocates up to three recommended attractions per day and orders stops by approximate geographic proximity.
- **Budget overview:** estimates attraction entry costs for the selected number of travellers and shows the remaining budget after those costs.
- **CSV export:** download the generated itinerary.
- **Nearby places:** search OpenStreetMap for mapped hotels or restaurants around a selected attraction, with a map and distance estimates.

## Technology

Python · Streamlit · Pandas · Scikit-learn · Requests · OpenStreetMap Overpass API

## Tests

Run the included logic tests:

```bash
python -m unittest discover -s tests -v
```

## Important data and planning limitations

The included CSV is a small illustrative starter dataset. Ratings and estimated entry fees are not verified live values. Nearby listings are community-mapped and do not confirm availability, current prices, ratings, or opening hours. Itinerary transfer times are indicative rather than live road-route estimates. The budget summary covers attraction entry costs only; it does not include flights, accommodation, food, or transport.

For a public production deployment, replace the starter dataset with a properly licensed, verified source; add appropriate API resilience and caching; and configure deployment secrets and usage limits where required.
