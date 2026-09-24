"""
One-time migration script: db.json -> travel.db (SQLite)

Run this once:
    python migrate_to_sqlite.py

It reads db.json and creates travel.db with 4 tables:
    users, flights, hotels, attractions

Nested fields (cabin_classes, amenities, tags, preferences, search_history)
are stored as JSON text columns, since SQLite has no native nested type.
main.py's db_helpers.py will parse them back into dicts/lists when reading,
so the rest of the app never notices the difference.
"""
import json
import sqlite3
import os

JSON_PATH = "db.json"
SQLITE_PATH = "travel.db"


def migrate():
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"'{JSON_PATH}' not found next to this script.")

    with open(JSON_PATH, "r") as f:
        db = json.load(f)

    if os.path.exists(SQLITE_PATH):
        os.remove(SQLITE_PATH)  # fresh build every time you run migration

    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()

    # ── users table ────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE users (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            home_city TEXT,
            preferences TEXT,       -- JSON
            search_history TEXT     -- JSON list
        )
    """)

    for user_id, u in db.items():
        if user_id in ("flights", "hotels", "attractions"):
            continue
        cur.execute(
            "INSERT INTO users (user_id, name, home_city, preferences, search_history) VALUES (?, ?, ?, ?, ?)",
            (
                user_id,
                u.get("name"),
                u.get("home_city"),
                json.dumps(u.get("preferences", {})),
                json.dumps(u.get("search_history", [])),
            ),
        )

    # ── flights table ───────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE flights (
            id TEXT PRIMARY KEY,
            airline TEXT,
            flight_no TEXT,
            origin TEXT,
            destination TEXT,
            departure TEXT,
            arrival TEXT,
            duration TEXT,
            stops TEXT,
            rating REAL,
            cabin_classes TEXT,   -- JSON
            amenities TEXT        -- JSON list
        )
    """)
    for f in db.get("flights", []):
        cur.execute(
            """INSERT INTO flights
               (id, airline, flight_no, origin, destination, departure, arrival,
                duration, stops, rating, cabin_classes, amenities)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f["id"], f["airline"], f["flight_no"], f["origin"], f["destination"],
                f["departure"], f["arrival"], f["duration"], f["stops"], f["rating"],
                json.dumps(f.get("cabin_classes", {})),
                json.dumps(f.get("amenities", [])),
            ),
        )

    # ── hotels table ────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE hotels (
            id TEXT PRIMARY KEY,
            name TEXT,
            stars INTEGER,
            area TEXT,
            city TEXT,
            price_per_night_usd REAL,
            price_per_night_inr REAL,
            rating REAL,
            vegetarian_friendly INTEGER,
            description TEXT,
            amenities TEXT,   -- JSON list
            tags TEXT         -- JSON list
        )
    """)
    for h in db.get("hotels", []):
        cur.execute(
            """INSERT INTO hotels
               (id, name, stars, area, city, price_per_night_usd, price_per_night_inr,
                rating, vegetarian_friendly, description, amenities, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                h["id"], h["name"], h["stars"], h["area"], h.get("city", ""),
                h["price_per_night_usd"], h["price_per_night_inr"], h["rating"],
                1 if h.get("vegetarian_friendly") else 0,
                h.get("description", ""),
                json.dumps(h.get("amenities", [])),
                json.dumps(h.get("tags", [])),
            ),
        )

    # ── attractions table ───────────────────────────────────────────
    cur.execute("""
        CREATE TABLE attractions (
            id TEXT PRIMARY KEY,
            name TEXT,
            category TEXT,
            area TEXT,
            city TEXT,
            entry_fee_usd REAL,
            best_time TEXT,
            duration TEXT,
            vegetarian_food_nearby INTEGER,
            description TEXT,
            tags TEXT   -- JSON list
        )
    """)
    for a in db.get("attractions", []):
        cur.execute(
            """INSERT INTO attractions
               (id, name, category, area, city, entry_fee_usd, best_time, duration,
                vegetarian_food_nearby, description, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                a["id"], a["name"], a["category"], a["area"], a.get("city", ""),
                a["entry_fee_usd"], a["best_time"], a["duration"],
                1 if a.get("vegetarian_food_nearby") else 0,
                a.get("description", ""),
                json.dumps(a.get("tags", [])),
            ),
        )

    conn.commit()
    conn.close()
    print(f"✅ Migration complete → {SQLITE_PATH}")
    print(f"   Users: {len(db) - 3} | Flights: {len(db.get('flights', []))} | "
          f"Hotels: {len(db.get('hotels', []))} | Attractions: {len(db.get('attractions', []))}")


if __name__ == "__main__":
    migrate()



    
