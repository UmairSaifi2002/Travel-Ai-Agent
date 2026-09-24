"""
Drop-in replacement for the old db.json-based helpers in main.py.

load_db() rebuilds the EXACT same dict shape that db.json used to give:
    {
        "<user_id>": {...},
        "flights_DEL_DXB": [...],
        "hotels_DXB": [...],
        "attractions_DXB": [...],
    }

So none of your existing tool functions (search_flights, search_hotels,
explore_attractions, build_trip_plan, get_flight_details, get_hotel_details)
need to change at all — they keep calling db.get("flights_DEL_DXB") etc.
"""
import json
import os
import sqlite3

DB_PATH = "travel.db"


def _connect():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"'{DB_PATH}' not found! Run `python migrate_to_sqlite.py` first."
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_db() -> dict:
    conn = _connect()
    cur = conn.cursor()
    result = {}

    # users
    cur.execute("SELECT * FROM users")
    for row in cur.fetchall():
        result[row["user_id"]] = {
            "name": row["name"],
            "home_city": row["home_city"],
            "preferences": json.loads(row["preferences"] or "{}"),
            "search_history": json.loads(row["search_history"] or "[]"),
        }

    # flights
    cur.execute("SELECT * FROM flights")
    flights = []
    for row in cur.fetchall():
        flights.append({
            "id": row["id"],
            "airline": row["airline"],
            "flight_no": row["flight_no"],
            "origin": row["origin"],
            "destination": row["destination"],
            "departure": row["departure"],
            "arrival": row["arrival"],
            "duration": row["duration"],
            "stops": row["stops"],
            "rating": row["rating"],
            "cabin_classes": json.loads(row["cabin_classes"] or "{}"),
            "amenities": json.loads(row["amenities"] or "[]"),
        })
    result["flights"] = flights

    # hotels
    cur.execute("SELECT * FROM hotels")
    hotels = []
    for row in cur.fetchall():
        hotels.append({
            "id": row["id"],
            "name": row["name"],
            "stars": row["stars"],
            "area": row["area"],
            "city": row["city"],
            "price_per_night_usd": row["price_per_night_usd"],
            "price_per_night_inr": row["price_per_night_inr"],
            "rating": row["rating"],
            "vegetarian_friendly": bool(row["vegetarian_friendly"]),
            "description": row["description"],
            "amenities": json.loads(row["amenities"] or "[]"),
            "tags": json.loads(row["tags"] or "[]"),
        })
    result["hotels"] = hotels

    # attractions
    cur.execute("SELECT * FROM attractions")
    attractions = []
    for row in cur.fetchall():
        attractions.append({
            "id": row["id"],
            "name": row["name"],
            "category": row["category"],
            "area": row["area"],
            "city": row["city"],
            "entry_fee_usd": row["entry_fee_usd"],
            "best_time": row["best_time"],
            "duration": row["duration"],
            "vegetarian_food_nearby": bool(row["vegetarian_food_nearby"]),
            "description": row["description"],
            "tags": json.loads(row["tags"] or "[]"),
        })
    result["attractions"] = attractions

    conn.close()
    return result


def load_user_profile(user_id: str) -> dict:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {}
    return {
        "name": row["name"],
        "home_city": row["home_city"],
        "preferences": json.loads(row["preferences"] or "{}"),
        "search_history": json.loads(row["search_history"] or "[]"),
    }


def log_search_to_db(user_id: str, search_entry: str):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT search_history FROM users WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        return
    history = json.loads(row["search_history"] or "[]")
    history.append(search_entry)
    cur.execute(
        "UPDATE users SET search_history = ? WHERE user_id = ?",
        (json.dumps(history), user_id),
    )
    conn.commit()
    conn.close()



    
