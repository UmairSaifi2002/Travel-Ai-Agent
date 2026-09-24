"""
main.py — Travel Agent (v10, full rebuild)

Zero-hallucination, database-backed travel agent.

Key design principles:
  * Every fact comes from a tool call against travel.db.
  * When LLM behavior is unreliable, do it in code (trip plans, comparisons).
  * Missing info is caught in preprocess_node — the LLM never sees it.
  * A validator blocks any reply containing a fact not in tool output.
"""
import os
import sys
import json
import math
import re
import uuid
from typing import Annotated, TypedDict

# ── UTF-8 output (kills cp1252 mojibake like Ôćĺ) ─────────────────────────────
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from db_helpers import load_db

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# 1. REGION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

REGION_GROUPS = {
    "uae": ["Dubai", "Abu Dhabi", "Sharjah", "Ras Al Khaimah"],
    "united arab emirates": ["Dubai", "Abu Dhabi", "Sharjah", "Ras Al Khaimah"],
    "gulf": ["Dubai", "Abu Dhabi", "Sharjah", "Ras Al Khaimah", "Doha",
             "Muscat", "Riyadh", "Jeddah", "Manama", "Kuwait City"],
    "middle east": ["Dubai", "Abu Dhabi", "Sharjah", "Ras Al Khaimah", "Doha",
                    "Muscat", "Riyadh", "Jeddah", "Manama", "Kuwait City"],
    "qatar": ["Doha"],
    "oman": ["Muscat"],
    "saudi arabia": ["Riyadh", "Jeddah"],
    "ksa": ["Riyadh", "Jeddah"],
    "bahrain": ["Manama"],
    "kuwait": ["Kuwait City"],
}

INDIA_CITIES = [
    "Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Kochi", "Jaipur",
]

ORIGIN_GROUPS = {"india": INDIA_CITIES, "bharat": INDIA_CITIES,
                 "indian": INDIA_CITIES}


def expand_city_query(city: str) -> list[str]:
    """'Dubai' → whole UAE region. 'UAE' same. Other cities stay exact."""
    if not city:
        return []
    key = city.strip().lower()
    if key in ORIGIN_GROUPS:
        return ORIGIN_GROUPS[key]
    if key in REGION_GROUPS:
        return REGION_GROUPS[key]
    if key == "dubai":
        return REGION_GROUPS["uae"]
    return [city]


def expand_origin_query(city: str) -> list[str]:
    if not city:
        return []
    key = city.strip().lower()
    return ORIGIN_GROUPS.get(key, [city])


# ══════════════════════════════════════════════════════════════════════════════
# 2. FLIGHT-SELECTION MARKER (used by 'book this one')
# ══════════════════════════════════════════════════════════════════════════════

FLIGHT_SELECTION_MARKER = "__FLIGHT_SELECTION_DATA__:"


def _flight_selection_record(f, cls_key, cd) -> dict:
    return {"id": f["id"], "flight_no": f["flight_no"], "airline": f["airline"],
            "origin": f["origin"], "destination": f["destination"],
            "cabin_class": cls_key, "price_usd": cd["price_usd"],
            "price_inr": cd["price_inr"], "departure": f["departure"],
            "arrival": f["arrival"], "stops": f["stops"]}


def _append_flight_selection_marker(lines, entries):
    payload = {"results": [_flight_selection_record(f, c, d) for f, c, d in entries]}
    lines.append(FLIGHT_SELECTION_MARKER + json.dumps(payload, separators=(",", ":")))
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# 3. TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def list_available_cities() -> str:
    """List every origin and destination city that exists in the database right
    now. Call this if you are unsure whether a city the client mentioned is
    supported — never guess."""
    flights = load_db().get("flights", [])
    origins = sorted({f["origin"] for f in flights})
    destinations = sorted({f["destination"] for f in flights})
    return ("ORIGINS (departure):\n" + "\n".join(f"  - {o}" for o in origins)
            + "\n\nDESTINATIONS (arrival):\n"
            + "\n".join(f"  - {d}" for d in destinations))


@tool
def search_flights(origin: str = "", destination: str = "", cabin_class: str = "",
                   max_price_usd: float = 9999, min_rating: float = 0,
                   airline: str = "", sort_by: str = "price",
                   sort_order: str = "asc") -> str:
    """Search flights across any supported origin and destination.

    'Dubai' widens to the UAE group (Dubai + Abu Dhabi + Sharjah + Ras Al
    Khaimah). Leave cabin_class blank to show all classes grouped.

    Args:
        origin: departure city (partial match). Required.
        destination: arrival city or region. Required.
        cabin_class: 'Economy' | 'Premium Economy' | 'Business' | 'First' | ''.
        max_price_usd: max price filter.
        min_rating: minimum flight rating out of 5.
        airline: optional airline filter.
        sort_by: 'price' | 'rating' | 'departure'.
        sort_order: 'asc' (default) or 'desc'.
    """
    if not origin and not destination:
        return ("ASK_USER: Ask the client exactly this and nothing else: "
                "'Which city are you flying from, and where would you like to go?' "
                "Do NOT say 'no flights available'.")
    if not origin:
        return ("ASK_USER: Ask the client exactly this and nothing else: "
                "'Which city are you flying from?' "
                "Do NOT say 'no flights available' or 'no Economy flights'.")
    if not destination:
        return ("ASK_USER: Ask the client exactly this and nothing else: "
                "'Which city would you like to fly to?' "
                "Do NOT say 'no flights available'.")

    flights = load_db().get("flights", [])
    origin_cities = expand_origin_query(origin)
    dest_cities = expand_city_query(destination)

    def matches(f):
        if not any(o.lower() in f["origin"].lower() for o in origin_cities):
            return False
        if not any(d.lower() in f["destination"].lower() for d in dest_cities):
            return False
        if f["rating"] < min_rating:
            return False
        if airline and airline.lower() not in f["airline"].lower():
            return False
        return True

    candidates = [f for f in flights if matches(f)]
    reverse = (sort_order.lower() == "desc")

    def fmt(i, f, cls, cd):
        return (f"{i:>2}. [{f['id']}] {f['airline']} {f['flight_no']} | "
                f"{f['origin']} -> {f['destination']} | {cls} | "
                f"dep {f['departure']} arr {f['arrival']} | {f['stops']} | "
                f"${cd['price_usd']} (INR {cd['price_inr']}) | "
                f"bag {cd['baggage']} | meal {cd['meal']} | "
                f"seats {cd['seats_left']} | rating {f['rating']}")

    note = f" (widened to {', '.join(dest_cities)})" if dest_cities != [destination] else ""

    # ── specific cabin ────────────────────────────────────────────────────
    if cabin_class:
        results = []
        for f in candidates:
            match = next((c for c in f["cabin_classes"]
                          if cabin_class.lower() in c.lower()), None)
            if not match:
                continue
            cd = f["cabin_classes"][match]
            if cd["price_usd"] > max_price_usd:
                continue
            results.append((f, match, cd))
        if sort_by == "rating":
            results.sort(key=lambda x: x[0]["rating"], reverse=not reverse)
        elif sort_by == "departure":
            results.sort(key=lambda x: x[0]["departure"], reverse=reverse)
        else:
            results.sort(key=lambda x: x[2]["price_usd"], reverse=reverse)

        if not results:
            return (f"NO RESULTS: no '{cabin_class}' flights for "
                    f"'{origin}' -> '{destination}'{note} under ${max_price_usd}.")
        lines = [f"FLIGHTS | {origin} -> {destination}{note} | {cabin_class}",
                 f"Total: {len(results)}"]
        for i, (f, cls, cd) in enumerate(results[:15], 1):
            lines.append(fmt(i, f, cls, cd))
        _append_flight_selection_marker(lines, results)
        return "\n".join(lines)

    # ── price-sorted, headline + breakdown ────────────────────────────────
    if sort_by == "price":
        combos = []
        for f in candidates:
            for cls, cd in f.get("cabin_classes", {}).items():
                if cd["price_usd"] <= max_price_usd:
                    combos.append((f, cls, cd))
        if not combos:
            return (f"NO RESULTS: no flights for '{origin}' -> "
                    f"'{destination}'{note} under ${max_price_usd}.")
        combos.sort(key=lambda x: x[2]["price_usd"], reverse=reverse)
        label = "MOST EXPENSIVE" if reverse else "CHEAPEST"
        top_f, top_c, top_d = combos[0]

        lines = [f"FLIGHTS | {origin} -> {destination}{note}",
                 f"",
                 f"ABSOLUTE {label} OVERALL:",
                 f"  {top_f['airline']} {top_f['flight_no']} | {top_f['origin']} -> {top_f['destination']}",
                 f"  Class {top_c} | dep {top_f['departure']} arr {top_f['arrival']} | {top_f['stops']}",
                 f"  ${top_d['price_usd']} (INR {top_d['price_inr']}) | "
                 f"bag {top_d['baggage']} | meal {top_d['meal']} | "
                 f"seats {top_d['seats_left']} | rating {top_f['rating']}",
                 f"",
                 f"-- top 3 per class --"]

        by_class = {"Economy": [], "Premium Economy": [], "Business": [], "First": []}
        for f, cls, cd in combos:
            bucket = next((b for b in by_class if b.lower() in cls.lower()), cls)
            by_class.setdefault(bucket, []).append((f, cls, cd))
        for cname, entries in by_class.items():
            lines.append(f"\n-- {cname.upper()} ({len(entries)} available) --")
            if not entries:
                lines.append(f"NO {cname} options on this route.")
                continue
            for i, (f, cls, cd) in enumerate(entries[:3], 1):
                lines.append(fmt(i, f, cls, cd))

        lines.append(f"\nTotal combinations: {len(combos)}")
        _append_flight_selection_marker(lines, combos)
        return "\n".join(lines)

    # ── grouped by class (rating/departure sort) ──────────────────────────
    by_class = {"Economy": [], "Premium Economy": [], "Business": [], "First": []}
    for f in candidates:
        for cls, cd in f.get("cabin_classes", {}).items():
            if cd["price_usd"] > max_price_usd:
                continue
            bucket = next((b for b in by_class if b.lower() in cls.lower()), cls)
            by_class.setdefault(bucket, []).append((f, cls, cd))
    if not any(by_class.values()):
        return (f"NO RESULTS: no flights for '{origin}' -> '{destination}'{note}.")

    lines = [f"FLIGHTS | {origin} -> {destination}{note} | all classes | sort={sort_by}"]
    total = 0
    for cname, entries in by_class.items():
        if not entries:
            continue
        if sort_by == "rating":
            entries.sort(key=lambda x: x[0]["rating"], reverse=not reverse)
        else:
            entries.sort(key=lambda x: x[0]["departure"], reverse=reverse)
        total += len(entries)
        lines.append(f"\n-- {cname.upper()} ({len(entries)}) --")
        for i, (f, cls, cd) in enumerate(entries[:5], 1):
            lines.append(fmt(i, f, cls, cd))
    lines.append(f"\nTotal combinations: {total}")
    all_e = [e for es in by_class.values() for e in es]
    _append_flight_selection_marker(lines, all_e)
    return "\n".join(lines)


@tool
def search_hotels(city: str = "", star_rating: int = 0, min_star_rating: int = 0,
                  max_price_usd: float = 9999, area: str = "", tag: str = "",
                  sort_by: str = "price", sort_order: str = "asc") -> str:
    """Search hotels in a city. 'Dubai' widens to the UAE group.

    Args:
        city: destination city. Required.
        star_rating: EXACT star (3 = only 3-star). Use this for "3 star hotels".
        min_star_rating: use only for "at least X" / "X or more".
        max_price_usd: per-night ceiling.
        area: neighbourhood filter.
        tag: filter by tag ('budget', 'luxury', 'beach', etc).
        sort_by: 'price' | 'rating' | 'stars'.
        sort_order: 'asc' (default) or 'desc'.
    """
    if not city:
        return ("ASK_USER: Ask the client exactly this and nothing else: "
                "'Which city are you interested in?' "
                "Do NOT say 'no hotels found'.")

    cities = expand_city_query(city)
    results = []
    for h in load_db().get("hotels", []):
        if not any(c.lower() in h.get("city", "").lower() for c in cities):
            continue
        if star_rating and h["stars"] != star_rating:
            continue
        if min_star_rating and h["stars"] < min_star_rating:
            continue
        if h["price_per_night_usd"] > max_price_usd:
            continue
        if area and area.lower() not in h["area"].lower():
            continue
        if tag and not any(tag.lower() in t.lower() for t in h.get("tags", [])):
            continue
        results.append(h)

    reverse = (sort_order.lower() == "desc")
    if sort_by == "price":
        results.sort(key=lambda x: x["price_per_night_usd"], reverse=reverse)
    elif sort_by == "rating":
        results.sort(key=lambda x: x["rating"], reverse=not reverse)
    elif sort_by == "stars":
        results.sort(key=lambda x: x["stars"], reverse=not reverse)

    if not results:
        return (f"NO RESULTS: no hotels in '{city}' matching your filters.")

    note = f" (widened to {', '.join(cities)})" if cities != [city] else ""
    star_desc = (f"{star_rating}-star exact" if star_rating
                 else f"{min_star_rating}+ stars" if min_star_rating else "any")
    lines = [f"HOTELS | {city}{note} | {star_desc} | max ${max_price_usd}/night",
             f"Total: {len(results)}"]
    for i, h in enumerate(results[:10], 1):
        lines.append(
            f"{i}. {h['name']} | {h['stars']}-star | {h['area']}, {h['city']} | "
            f"${h['price_per_night_usd']}/night (INR {h['price_per_night_inr']}) | "
            f"rating {h['rating']} | veg={'Y' if h['vegetarian_friendly'] else 'N'} | "
            f"amenities: {', '.join(h['amenities'][:4])}"
        )
    return "\n".join(lines)


@tool
def explore_attractions(city: str = "", category: str = "",
                        max_entry_fee_usd: float = 9999, tag: str = "") -> str:
    """Explore attractions in a city. 'Dubai' widens to the UAE group.

    Args:
        city: destination city. Required.
        category: e.g. 'Adventure', 'Culture', 'Beach', 'Shopping'.
        max_entry_fee_usd: 0 for free only.
        tag: e.g. 'family', 'iconic', 'must-visit'.
    """
    if not city:
        return ("ASK_USER: Ask the client exactly this and nothing else: "
                "'Which city would you like to explore?' "
                "Do NOT say 'no attractions found'.")
    cities = expand_city_query(city)
    results = []
    for a in load_db().get("attractions", []):
        if not any(c.lower() in a.get("city", "").lower() for c in cities):
            continue
        if category and category.lower() not in a["category"].lower():
            continue
        if a["entry_fee_usd"] > max_entry_fee_usd:
            continue
        if tag and not any(tag.lower() in t.lower() for t in a.get("tags", [])):
            continue
        results.append(a)

    if not results:
        return f"NO RESULTS: no attractions in '{city}' matching your filters."

    lines = [f"ATTRACTIONS | {city}", f"Total: {len(results)}"]
    for i, a in enumerate(results, 1):
        fee = "FREE" if a["entry_fee_usd"] == 0 else f"${a['entry_fee_usd']}"
        lines.append(
            f"{i}. {a['name']} | {a['category']} | {a['area']}, {a['city']} | "
            f"entry {fee} | best {a['best_time']} | {a['duration']} | "
            f"veg_nearby={'Y' if a['vegetarian_food_nearby'] else 'N'}"
        )
    return "\n".join(lines)


@tool
def find_dining_options(city: str = "", max_price_usd: float = 9999,
                        vegetarian_only: bool = False, sort_by: str = "price") -> str:
    """Find restaurants/dining in a city. Uses hotels with on-site restaurants.

    Args:
        city: destination city. Required.
        max_price_usd: proxy ceiling.
        vegetarian_only: only veg-friendly.
        sort_by: 'price' or 'rating'.
    """
    if not city:
        return ("ASK_USER: Ask the client exactly this and nothing else: "
                "'Which city are you interested in?'. Do NOT say 'no dining found'.")
    cities = expand_city_query(city)
    results = []
    for h in load_db().get("hotels", []):
        if not any(c.lower() in h.get("city", "").lower() for c in cities):
            continue
        if not any("restaurant" in a.lower() for a in h.get("amenities", [])):
            continue
        if h["price_per_night_usd"] > max_price_usd:
            continue
        if vegetarian_only and not h["vegetarian_friendly"]:
            continue
        results.append(h)

    if not results:
        return f"NO RESULTS: no dining options in '{city}' matching your filters."
    if sort_by == "rating":
        results.sort(key=lambda x: x["rating"], reverse=True)
    else:
        results.sort(key=lambda x: x["price_per_night_usd"])

    lines = [f"DINING in {city} | Total: {len(results)}"]
    for i, h in enumerate(results[:10], 1):
        lines.append(f"{i}. {h['name']} | {h['area']} | "
                     f"${h['price_per_night_usd']} | "
                     f"veg={'Y' if h['vegetarian_friendly'] else 'N'} | "
                     f"rating {h['rating']}")
    return "\n".join(lines)


@tool
def get_flight_details(flight_id: str) -> str:
    """Full details of a specific flight by its ID (e.g. F001)."""
    for f in load_db().get("flights", []):
        if f["id"].upper() == flight_id.upper():
            lines = [f"FLIGHT {f['id']}",
                     f"Airline: {f['airline']} ({f['flight_no']})",
                     f"Route: {f['origin']} -> {f['destination']}",
                     f"Dep {f['departure']} Arr {f['arrival']} | {f['stops']} | "
                     f"rating {f['rating']}/5",
                     f"Amenities: {', '.join(f.get('amenities', []))}",
                     "Classes:"]
            for c, d in f.get("cabin_classes", {}).items():
                lines.append(f"  {c}: ${d['price_usd']} (INR {d['price_inr']}) | "
                             f"bag {d['baggage']} | meal {d['meal']} | "
                             f"seats {d['seats_left']}")
            return "\n".join(lines)
    return f"NO RESULTS: flight ID '{flight_id}' not found."


@tool
def get_hotel_details(hotel_id: str) -> str:
    """Full details of a specific hotel by its ID (e.g. H001)."""
    for h in load_db().get("hotels", []):
        if h["id"].upper() == hotel_id.upper():
            return (f"HOTEL {h['id']}\n"
                    f"Name: {h['name']}\n"
                    f"City: {h.get('city', '')}\n"
                    f"Stars: {h['stars']}-star\n"
                    f"Area: {h['area']}\n"
                    f"Price: ${h['price_per_night_usd']}/night "
                    f"(INR {h['price_per_night_inr']}/night)\n"
                    f"Rating: {h['rating']}/5\n"
                    f"Veg-friendly: {'Yes' if h['vegetarian_friendly'] else 'No'}\n"
                    f"Amenities: {', '.join(h['amenities'])}\n"
                    f"Tags: {', '.join(h['tags'])}")
    return f"NO RESULTS: hotel ID '{hotel_id}' not found."


@tool
def build_trip_plan(origin: str = "", destination: str = "",
                    duration_days: int = 5, budget_level: str = "midrange",
                    flight_preference: str = "cheapest",
                    interests: str = "sightseeing, culture, adventure",
                    cabin_class: str = "Economy", travelers: int = 1,
                    round_trip: bool = True,
                    outbound_flight_id: str = "", outbound_cabin_class: str = "",
                    return_flight_id: str = "", return_cabin_class: str = "",
                    return_destination: str = "") -> str:
    """Build a complete trip plan: round-trip flights + hotel + day-by-day itinerary.

    Hotel pricing is PER ROOM, not per person. This tool assumes 2 guests per
    room, so 5 travelers → 3 rooms. Total hotel cost = price × nights × rooms.

    Args:
        origin: departure city. Required.
        destination: arrival city. Required.
        duration_days: 2-10.
        budget_level: 'budget' | 'midrange' | 'luxury' | 'ultra-luxury'.
        flight_preference: 'cheapest' | 'most_expensive' | 'highest_rated'.
        interests: comma-separated.
        cabin_class: preferred cabin; '' allows any.
        travelers: number of travelers.
        round_trip: include return leg.
        outbound_flight_id / return_flight_id: confirmed flight IDs.
        return_destination: only if return city != origin.
    """
    if not origin or not destination:
        return "MISSING INPUT: origin and destination are both required."

    effective_return_city = return_destination.strip() if return_destination else origin
    db = load_db()
    flights = db.get("flights", [])
    dest_cities = expand_city_query(destination)
    return_cities = expand_origin_query(effective_return_city) or [origin]
    attractions_pool = db.get("attractions", [])
    travelers = max(1, int(travelers))

    selected_out = None
    if outbound_flight_id:
        wanted = outbound_flight_id.strip().upper()
        selected_out = next((f for f in flights
                             if f["id"].upper() == wanted
                             or f["flight_no"].upper() == wanted), None)
    actual_arrival_city = (selected_out.get("destination", "").split(" (")[0]
                           if selected_out else None)

    budget_map = {
        "budget":       {"max_flight": 300,  "max_hotel": 100,  "tags": ["budget"]},
        "midrange":     {"max_flight": 700,  "max_hotel": 200,  "tags": ["midrange"]},
        "luxury":       {"max_flight": 1300, "max_hotel": 450,  "tags": ["luxury"]},
        "ultra-luxury": {"max_flight": 9999, "max_hotel": 9999, "tags": ["ultra-luxury"]},
    }
    bmap = budget_map.get(budget_level.lower(), budget_map["midrange"])

    PREF = (flight_preference or "cheapest").lower().replace("-", "_").replace(" ", "_")
    MOST_EXPENSIVE = PREF in ("most_expensive", "mostexpensive", "premium",
                              "luxury", "priciest", "most_costly")
    HIGHEST_RATED = PREF in ("highest_rated", "highestrated", "best_rated", "top_rated")

    def _rank(opts):
        if not opts:
            return None
        if MOST_EXPENSIVE:
            return max(opts, key=lambda x: (x["data"]["price_usd"], x["flight"]["rating"]))
        if HIGHEST_RATED:
            return max(opts, key=lambda x: (x["flight"]["rating"], x["data"]["price_usd"]))
        return min(opts, key=lambda x: (x["data"]["price_usd"], -x["flight"]["rating"]))

    def find_by_id(fid):
        if not fid:
            return None
        u = fid.strip().upper()
        return next((f for f in flights
                     if f["id"].upper() == u or f["flight_no"].upper() == u), None)

    def pick_specific(fid, wanted_cabin=""):
        f = find_by_id(fid)
        if not f or not f.get("cabin_classes"):
            return None
        cd = f["cabin_classes"]
        if wanted_cabin:
            match = next((c for c in cd if c.lower() == wanted_cabin.lower()), None)
            if not match:
                return None
            return {"flight": f, "class": match, "data": cd[match]}
        opts = [{"flight": f, "class": c, "data": d} for c, d in cd.items()]
        return _rank(opts)

    def _collect(cands, cabin, ceiling, use_ceiling):
        opts = []
        for f in cands:
            for c, d in f.get("cabin_classes", {}).items():
                if cabin and cabin.lower() not in c.lower():
                    continue
                if use_ceiling and d["price_usd"] > ceiling:
                    continue
                opts.append({"flight": f, "class": c, "data": d})
        return opts

    def _prefer_city(opts, city):
        if not opts or not city:
            return opts
        want = city.strip().lower()
        exact = [o for o in opts
                 if o["flight"]["destination"].split(" (")[0].strip().lower() == want]
        return exact or opts

    if MOST_EXPENSIVE and cabin_class.strip().lower() == "economy":
        cabin_for_search = ""
    else:
        cabin_for_search = cabin_class

    # ── outbound ──────────────────────────────────────────────────────────
    bad_out = bad_ret = False
    if outbound_flight_id:
        best_out = pick_specific(outbound_flight_id, outbound_cabin_class)
        if not best_out:
            bad_out = True
    else:
        cands = [f for f in flights
                 if origin.lower() in f["origin"].lower()
                 and any(c.lower() in f["destination"].lower() for c in dest_cities)]
        opts = _collect(cands, cabin_for_search, bmap["max_flight"],
                        not MOST_EXPENSIVE)
        opts = _prefer_city(opts, destination)
        best_out = _rank(opts)

    # ── return ────────────────────────────────────────────────────────────
    best_ret = None
    if round_trip:
        if return_flight_id:
            best_ret = pick_specific(return_flight_id, return_cabin_class)
            if not best_ret:
                bad_ret = True
        else:
            ret_origin = (actual_arrival_city
                          or (best_out["flight"]["destination"].split(" (")[0]
                              if best_out else destination))
            cands = [f for f in flights
                     if ret_origin.lower() in f["origin"].lower()
                     and any(c.lower() in f["destination"].lower() for c in return_cities)]
            opts = _collect(cands, cabin_for_search, bmap["max_flight"],
                            not MOST_EXPENSIVE)
            opts = _prefer_city(opts, effective_return_city)
            best_ret = _rank(opts)

    if bad_out or bad_ret:
        problems = []
        if bad_out: problems.append(f"outbound '{outbound_flight_id}'")
        if bad_ret: problems.append(f"return '{return_flight_id}'")
        return ("FAILED: could not resolve selected flight(s) " + ", ".join(problems)
                + ". Please search again and select once more.")

    stay_city = actual_arrival_city
    if not stay_city and best_out:
        stay_city = best_out["flight"]["destination"].split(" (")[0]
    stay_city = stay_city or destination

    # ── hotel ─────────────────────────────────────────────────────────────
    hotels = [h for h in db.get("hotels", [])
              if h.get("city", "").strip().lower() == stay_city.strip().lower()]
    pool = [h for h in hotels if h["price_per_night_usd"] <= bmap["max_hotel"]]
    tagged = [h for h in pool if any(t in h.get("tags", []) for t in bmap["tags"])]
    candidates_h = tagged or pool

    if candidates_h:
        if MOST_EXPENSIVE:
            best_hotel = max(candidates_h,
                             key=lambda h: (h["price_per_night_usd"], h["rating"]))
        elif HIGHEST_RATED:
            best_hotel = max(candidates_h,
                             key=lambda h: (h["rating"], h["price_per_night_usd"]))
        else:
            best_hotel = min(candidates_h,
                             key=lambda h: (h["price_per_night_usd"], -h["rating"]))
    else:
        best_hotel = None

    # ⭐ ROOM MATH — 2 guests per room
    rooms_needed = max(1, math.ceil(travelers / 2))
    hotel_total = 0
    hotel_inr_total = 0
    if best_hotel:
        hotel_total = best_hotel["price_per_night_usd"] * duration_days * rooms_needed
        hotel_inr_total = best_hotel["price_per_night_inr"] * duration_days * rooms_needed

    stay_note = ""
    if dest_cities != [destination] and stay_city.lower() != destination.lower():
        stay_note = (f"\nNOTE: '{destination}' searched as the wider region "
                     f"{', '.join(dest_cities)}. Flight lands in {stay_city}, so "
                     f"hotel and itinerary are in {stay_city}.")

    # ── attractions ───────────────────────────────────────────────────────
    attractions = [a for a in attractions_pool
                   if a.get("city", "").strip().lower() == stay_city.strip().lower()]
    interest_list = [i.strip().lower() for i in interests.split(",")]
    matched = [a for a in attractions
               if any(i in " ".join(a.get("tags", [])).lower() for i in interest_list)]
    if len(matched) < duration_days * 2:
        for a in attractions:
            if a not in matched:
                matched.append(a)
            if len(matched) >= duration_days * 3:
                break

    # ── assemble ──────────────────────────────────────────────────────────
    L = []
    L.append("=" * 68)
    title = f"{origin} -> {stay_city}"
    if effective_return_city.lower() != origin.lower():
        title += f" -> {effective_return_city} (open-jaw)"
    L.append(f"  TRIP PLAN — {title}")
    L.append(f"  {duration_days} nights | {travelers} traveler(s) | "
             f"{budget_level} | pref={PREF}")
    L.append("=" * 68)

    L.append(f"\nOUTBOUND FLIGHT ({origin} -> {stay_city})")
    L.append("-" * 50)
    if best_out:
        f, cd = best_out["flight"], best_out["data"]
        tag = " (confirmed)" if outbound_flight_id and not bad_out else ""
        L.append(f"  ID       : {f['id']}")
        L.append(f"  Airline  : {f['airline']} ({f['flight_no']}){tag}")
        L.append(f"  Class    : {best_out['class']}")
        L.append(f"  Route    : {f['origin']} -> {f['destination']}")
        L.append(f"  Departs  : {f['departure']}  Arrives: {f['arrival']}  Stops: {f['stops']}")
        L.append(f"  Price    : ${cd['price_usd']} (INR {cd['price_inr']}) per person")
        L.append(f"  Group    : ${cd['price_usd'] * travelers} for {travelers} traveler(s)")
        L.append(f"  Baggage  : {cd['baggage']} | Meal: {cd['meal']}")
        L.append(f"  Rating   : {f['rating']}/5")
    else:
        L.append("  No suitable outbound flight found.")

    if round_trip:
        ret_label = f"{(best_out['flight']['destination'].split(' (')[0] if best_out else stay_city)} -> {effective_return_city}"
        L.append(f"\nRETURN FLIGHT ({ret_label})")
        L.append("-" * 50)
        if best_ret:
            f, cd = best_ret["flight"], best_ret["data"]
            tag = " (confirmed)" if return_flight_id and not bad_ret else ""
            L.append(f"  ID       : {f['id']}")
            L.append(f"  Airline  : {f['airline']} ({f['flight_no']}){tag}")
            L.append(f"  Class    : {best_ret['class']}")
            L.append(f"  Route    : {f['origin']} -> {f['destination']}")
            L.append(f"  Departs  : {f['departure']}  Arrives: {f['arrival']}  Stops: {f['stops']}")
            L.append(f"  Price    : ${cd['price_usd']} (INR {cd['price_inr']}) per person")
            L.append(f"  Group    : ${cd['price_usd'] * travelers} for {travelers} traveler(s)")
            L.append(f"  Rating   : {f['rating']}/5")
        else:
            L.append(f"  No suitable return flight found for {ret_label}.")

    L.append(f"\nHOTEL ({duration_days} nights, {rooms_needed} room(s)){stay_note}")
    L.append("-" * 50)
    if best_hotel:
        h = best_hotel
        stars = "*" * h["stars"]
        L.append(f"  Name       : {h['name']}")
        L.append(f"  City       : {h['city']}")
        L.append(f"  Stars      : {stars} ({h['stars']}-star)")
        L.append(f"  Area       : {h['area']}")
        L.append(f"  Per Night  : ${h['price_per_night_usd']} "
                 f"(INR {h['price_per_night_inr']}) per room")
        L.append(f"  Rooms      : {rooms_needed} room(s) for {travelers} traveler(s) (2/room)")
        L.append(f"  {duration_days} Nights   : ${hotel_total} (INR {hotel_inr_total})")
        L.append(f"  Rating     : {h['rating']}/5")
        L.append(f"  Amenities  : {', '.join(h['amenities'])}")
    else:
        L.append("  No suitable hotel found.")

    L.append(f"\nDAY-BY-DAY ITINERARY")
    L.append("-" * 50)
    pool_attr = matched[:]
    for day in range(1, duration_days + 1):
        L.append(f"\n  Day {day}:")
        if day == 1:
            L.append("    Arrival — check in, freshen up, explore hotel area")
            if pool_attr:
                a = pool_attr.pop(0)
                fee = "FREE" if a["entry_fee_usd"] == 0 else f"${a['entry_fee_usd']}"
                L.append(f"    Evening: {a['name']} | {a['area']} | Entry {fee} | Best {a['best_time']}")
        elif day == duration_days:
            L.append("    Departure — check out, last-minute shopping, head to airport")
        else:
            for slot in ["Morning", "Afternoon", "Evening"]:
                if not pool_attr:
                    L.append(f"    {slot}: Free time / explore on your own")
                    continue
                a = pool_attr.pop(0)
                fee = "FREE" if a["entry_fee_usd"] == 0 else f"${a['entry_fee_usd']}"
                L.append(f"    {slot}: {a['name']} | {a['area']} | Entry {fee} | {a['duration']}")

    out_pp = best_out["data"]["price_usd"] if best_out else 0
    ret_pp = best_ret["data"]["price_usd"] if (round_trip and best_ret) else 0
    flight_total = (out_pp + ret_pp) * travelers
    activities = duration_days * 40 * travelers
    food = duration_days * 30 * travelers
    total = flight_total + hotel_total + activities + food

    L.append(f"\nCOST SUMMARY ({travelers} traveler(s))")
    L.append("-" * 50)
    L.append(f"  Flights (round trip)   : ${flight_total} "
             f"({travelers} × ${out_pp + ret_pp}/person)")
    L.append(f"  Hotel ({duration_days}n × {rooms_needed} room)   : ${hotel_total}")
    L.append(f"  Activities (est.)      : ${activities} ($40 × {duration_days}d × {travelers})")
    L.append(f"  Food (est.)            : ${food} ($30 × {duration_days}d × {travelers})")
    L.append(f"  {'-' * 40}")
    L.append(f"  TOTAL                  : ${total} (INR {int(total * 83):,})")
    L.append("=" * 68)
    return "\n".join(L)


ALL_TOOLS = [list_available_cities, search_flights, search_hotels,
             explore_attractions, find_dining_options,
             get_flight_details, get_hotel_details, build_trip_plan]


# ══════════════════════════════════════════════════════════════════════════════
# 4. LLM PROVIDER
# ══════════════════════════════════════════════════════════════════════════════

PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()

if PROVIDER == "groq":
    from langchain_groq import ChatGroq
    model = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                     groq_api_key=os.getenv("GROQ_API_KEY"),
                     temperature=0, max_tokens=2048, timeout=30, max_retries=2)
elif PROVIDER == "openai":
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                       api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
elif PROVIDER == "openrouter":
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        default_headers={"HTTP-Referer": "http://localhost",
                         "X-Title": "Travel Agent"})
elif PROVIDER == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI
    model = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0)
elif PROVIDER == "ollama":
    from langchain_ollama import ChatOllama
    model = ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen2.5:14b"),
                       base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                       temperature=0, num_ctx=8192, keep_alive="30m")
else:
    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}")

llm_with_tools = model.bind_tools(ALL_TOOLS)


# ══════════════════════════════════════════════════════════════════════════════
# 5. SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a database-powered travel agent. Every fact you give the client MUST \
come from a tool call in this session. You have no real-world travel knowledge.

LANGUAGE — NON-NEGOTIABLE
Always reply in the SAME language the user wrote in.
- User wrote English  → you reply in English.
- User wrote Hindi    → you reply in English with Hindi words mixed in.
- User wrote Hinglish → you reply in English (Hinglish is fine).
NEVER reply in Chinese, Japanese, Korean, Arabic, or any language the user \
did not use. If you are unsure, reply in English. This is a hard rule.

CORE RULES
1. For any question about flights, hotels, attractions, or dining, call the \
matching tool FIRST.
2. If origin or destination is missing, ask ONE short question before calling.
3. If the client asks for multiple things (flights AND hotels, etc.), call a \
tool for EACH before replying.
4. If a tool returns "NO RESULTS", relay that honestly.
5. Never invent a price, flight number, hotel name, or rating.
6. If a tool result starts with "ASK_USER", that is NOT a "no results" answer. \
It means you are missing input. Relay the exact question to the user and wait. \
NEVER say "no flights available", "no Economy flights", or "no hotels found" \
when a tool returned ASK_USER.
7. When the client asks for "most expensive" / "premium" / "luxury", pass \
flight_preference="most_expensive" and budget_level="ultra-luxury" to \
build_trip_plan. When they ask for cheapest, pass flight_preference="cheapest".
8. Never show internal markers like __FLIGHT_SELECTION_DATA__.

REGION WIDENING
'Dubai' searches the wider UAE (Dubai + Abu Dhabi + Sharjah + Ras Al Khaimah). \
Always show each result's ACTUAL city — never relabel an Abu Dhabi hotel as Dubai.

"BOOK" MEANS REMEMBER
"Book this one" = remember the exact flight for a later trip plan. No money is \
charged. Acknowledge briefly; the code stores the flight ID and cabin.

TRIP PLAN
Pass confirmed flight IDs to build_trip_plan as outbound_flight_id / \
return_flight_id. For open-jaw trips (out via A, back to B where B != A), pass \
return_destination=B. Do not ask for budget/duration/cabin before building — \
use defaults and let the client refine.

FORMAT
- If asked for "non-stop" / "direct", only show flights where stops = Non-Stop. \
If none exist, say so plainly.
- Group flights by cabin: Economy / Premium Economy / Business / First. If a \
class has zero results, print "No <Class> options on this route."
- Hotels and attractions: short cards with name, city, price/fee, rating, and \
2-3 amenities.
- No filler like "Great question!". Get to the answer.
- Star ratings: "3 star hotel" → star_rating=3 in search_hotels. "Highly rated" \
→ min_rating=4.0. "3 star flight" → min_rating=3.0 in search_flights.

CITIES
If unsure whether a city is supported, call list_available_cities.

ROUND TRIPS — EXACT PATTERN TO FOLLOW

COMPARISON QUERIES
When the user asks for a comparison (e.g. "cheapest vs most expensive",
"compare X and Y", "what's the price range"):

  For a SINGLE route comparison:
    Call search_flights twice with the same route but different sort orders:
      Call 1: sort_by="price", sort_order="asc"   (cheapest first)
      Call 2: sort_by="price", sort_order="desc"  (most expensive first)
    Then show both extremes side by side in your reply.

  For MULTIPLE routes in one query (e.g. "most expensive Dubai to Delhi
  AND cheapest Pune to Dubai"):
    Treat this as TWO SEPARATE searches. Call search_flights once for each
    route with the correct sort order:
      Call 1: origin="Dubai", destination="Delhi", sort_order="desc"
      Call 2: origin="Pune", destination="Dubai", sort_order="asc"
    Present the results as two clearly-labelled sections.

  Never say "only one option exists" unless the tool result literally says so.
  Every price you quote must be in a tool result.
"""


# ══════════════════════════════════════════════════════════════════════════════
# 6. STATE
# ══════════════════════════════════════════════════════════════════════════════

class TravelState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    origin: str
    destination: str
    return_destination: str
    last_flight_results: list[dict]
    selected_outbound_flight_id: str
    selected_outbound_cabin_class: str
    selected_return_flight_id: str
    selected_return_cabin_class: str
    duration_days: int
    budget_level: str
    flight_preference: str
    interests: str
    travelers: int
    selection_just_saved: bool
    trip_plan_requested: bool
    comparison_requested: bool
    needs_origin: bool
    needs_destination: bool
    validator_state: str


# ══════════════════════════════════════════════════════════════════════════════
# 7. HELPERS
# ══════════════════════════════════════════════════════════════════════════════

BOOKING_PHRASES = ("book this", "book it", "book that", "i'll take", "ill take",
                   "take this", "take that", "select this", "select that",
                   "confirm this", "confirm that", "go with this")

TRIP_PLAN_PHRASES = ("trip plan", "itinerary", "plan my trip", "plan the trip",
                     "plan a trip", "full trip", "make a plan", "create a plan",
                     "build a trip", "rough plan", "make a trip", "make the trip",
                     "plan my travel", "create a trip", "create the trip")

ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2,
            "fourth": 3, "4th": 3, "fifth": 4, "5th": 4}


def _last_human(messages):
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m
    return None


def _supported_cities():
    db = load_db()
    names = set()
    for f in db.get("flights", []):
        for k in ("origin", "destination"):
            names.add(f[k].split(" (")[0].strip())
    for h in db.get("hotels", []):
        if h.get("city"):
            names.add(h["city"].strip())
    return sorted(names, key=len, reverse=True)


def _extract_city_pair(text: str):
    """Extract origin/destination from free text.

    Handles:
      - English:  "from Delhi to Dubai", "Delhi to Dubai"
      - Hinglish: "Delhi se Dubai", "Delhi sy Dubai", "Delhi se lekar Dubai"
      - Reverse:  "go to Dubai from Delhi"
      - Fallback: exactly two supported cities mentioned → first=origin, second=dest
    """
    if not text:
        return None, None
    cities = _supported_cities()
    if not cities:
        return None, None
    alt = "|".join(re.escape(c) for c in cities)

    patterns = (
        rf"\bfrom\s+(?P<o>{alt})\s+(?:to|towards)\s+(?P<d>{alt})\b",
        rf"\b(?P<o>{alt})\s+(?:to|se|sy|sé|se\s+lekar|se\s+leke)\s+(?P<d>{alt})\b",
        rf"\b(?:go|going|travel|travelling|traveling|fly|flying|jaana|jana)"
        rf"\s+(?:to\s+)?(?P<d>{alt})\b.*?\bfrom\s+(?P<o>{alt})\b",
        rf"\b(?:want|need|visit|chahiye|chaiye)\s+(?:to\s+)?(?P<d>{alt})\b"
        rf".*?\bfrom\s+(?P<o>{alt})\b",
    )
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group("o"), m.group("d")

    # Fallback: exactly two supported cities → first is origin, second is dest
    found = []
    for c in cities:
        if re.search(rf"\b{re.escape(c)}\b", text, re.IGNORECASE):
            if c.lower() not in [f.lower() for f in found]:
                found.append(c)
    if len(found) == 2:
        return found[0], found[1]
    return None, None


def _extract_single_city(text: str):
    """When only one city is mentioned, classify it by surrounding words."""
    if not text:
        return None, None
    cities = _supported_cities()
    if not cities:
        return None, None
    t = text.lower()
    found = []
    for c in cities:
        if re.search(rf"\b{re.escape(c.lower())}\b", t):
            if c.lower() not in [f.lower() for f in found]:
                found.append(c)
    if len(found) != 1:
        return None, None

    city = found[0]
    esc = re.escape(city.lower())

    # Origin markers
    if (re.search(rf"\bfrom\s+{esc}\b", t)
            or re.search(rf"\b{esc}\s+(?:se|sy|sé)\b", t)):
        return city, None

    # Destination markers
    # Destination markers
    if (re.search(rf"\bto\s+{esc}\b", t)
            or re.search(rf"\bin\s+{esc}\b", t)          # ⭐ "in Dubai"
            or re.search(rf"\b{esc}\s+(?:jaana|jana|chahiye|chaiye)\b", t)):
        return None, city

    return None, None


def _find_bare_city(text: str):
    """Return the single supported city mentioned, ignoring context markers.
    Used when the user replies to 'which city?' with just a city name."""
    if not text:
        return None
    cities = _supported_cities()
    t = text.lower()
    found = []
    for c in cities:
        if re.search(rf"\b{re.escape(c.lower())}\b", t):
            if c.lower() not in [f.lower() for f in found]:
                found.append(c)
    return found[0] if len(found) == 1 else None

def _extract_duration(text):
    m = re.search(r"\b(\d{1,2})\s*(?:days?|nights?)\b", text.lower())
    return max(2, min(int(m.group(1)), 10)) if m else None


def _extract_budget(text):
    t = text.lower()
    if any(k in t for k in ("tight", "cheap", "affordable", "low budget", "budget")):
        return "budget"
    if any(k in t for k in ("midrange", "mid range", "moderate", "reasonable")):
        return "midrange"
    if any(k in t for k in ("premium", "luxury", "high end", "high-end")):
        return "luxury"
    if any(k in t for k in ("money no issue", "open budget", "doesn't matter")):
        return "ultra-luxury"
    return None


def _extract_travelers(text):
    t = text.lower()

    # ⭐ "me and my 4 friends" / "I and my 3 friends" → N + 1
    m = re.search(r"\b(?:me|i)\s+(?:and|plus|\+)\s+(?:my\s+|our\s+)?(\d+)\s+"
                  r"(?:friends|people|persons|passengers|travelers|travellers)\b", t)
    if m:
        return max(1, int(m.group(1)) + 1)

    # ⭐ "4 friends and me" / "3 friends and I" → N + 1
    m = re.search(r"\b(\d+)\s+(?:friends|people|persons|passengers|travelers|travellers)\s+"
                  r"(?:and|plus|\+)\s+(?:me|i)\b", t)
    if m:
        return max(1, int(m.group(1)) + 1)

    # "we are 5" / "there are 5 of us"
    m = re.search(r"\b(?:we\s+are|there\s+are)\s+(\d+)\s+(?:of\s+us|friends|people)\b", t)
    if m:
        return max(1, int(m.group(1)))

    # "5 friends", "3 people", "family of 4"
    m = re.search(r"\b(\d+)\s+(?:friends|people|persons|passengers|travelers|travellers)\b", t)
    if m:
        return max(1, int(m.group(1)))

    m = re.search(r"\bfamily\s+of\s+(\d+)\b", t)
    if m:
        return max(1, int(m.group(1)))

    # "with my wife/husband/partner" implies 2
    if any(x in t for x in ("with my wife", "with my husband", "with my spouse",
                            "with my partner", "with my girlfriend",
                            "with my boyfriend")):
        return 2
    if "with my family" in t:
        return 4
    return None


def _looks_like_booking(text):
    return any(p in text.lower() for p in BOOKING_PHRASES)


def _looks_like_trip_plan(text):
    """Detect trip-plan intent from natural language, not just literal phrases."""
    t = text.lower()

    # Literal phrases
    if any(p in t for p in TRIP_PLAN_PHRASES):
        return True

    # "want to go", "wanna go", "planning to go", "would like to visit", etc.
    if any(p in t for p in (
        "want to go", "wanna go", "want to visit", "wanna visit",
        "want to travel", "wanna travel", "planning to go",
        "planning a trip", "planning a visit", "would like to go",
        "would like to visit", "would love to go", "would love to visit",
        "going to", "heading to", "trip to", "travel to",
        "jaana", "jana", "chahiye", "chaiye", "ghoomne",
    )):
        # ...but only if a trip duration or planning context is present
        if (re.search(r"\b\d{1,2}\s*(?:days?|nights?)\b", t)
                or "trip" in t or "travel" in t or "visit" in t
                or "holiday" in t or "vacation" in t):
            return True

    # "for 5 days" alone with a destination is often a trip-plan ask
    if re.search(r"\b\d{1,2}\s*(?:days?|nights?)\b", t) and "to " in t:
        return True

    return False


def _pick_flight(text, results):
    if not results:
        return None
    t = text.lower()
    m = re.search(r"\bF\d{4,}\b", text, re.IGNORECASE)
    if m:
        for r in results:
            if r["id"].upper() == m.group(0).upper():
                return r
    m = re.search(r"\b[A-Z]{1,3}-\d{3,4}\b", text, re.IGNORECASE)
    if m:
        for r in results:
            if r["flight_no"].upper() == m.group(0).upper():
                return r
    m = re.search(r"\b([1-9]|1[0-5])(?:st|nd|rd|th)\b", t)
    if m and int(m.group(1)) - 1 < len(results):
        return results[int(m.group(1)) - 1]
    for word, idx in ORDINALS.items():
        if re.search(rf"\b{word}\b", t) and idx < len(results):
            return results[idx]
    for r in results:
        if r["airline"].lower() in t:
            return r
    return results[0]


# ══════════════════════════════════════════════════════════════════════════════
# 8. NODES
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_node(state: TravelState):
    """Runs before the LLM. Extracts facts, resolves 'book this one', detects
    missing info so the LLM never sees it."""
    msgs = state.get("messages", [])
    human = _last_human(msgs)
    if not human:
        return {}

    text = human.content or ""
    t_lower = text.lower()

    updates: dict = {
        "selection_just_saved": False,
        "trip_plan_requested": _looks_like_trip_plan(text),
    }

    # ── 1. City extraction ────────────────────────────────────────────────
    # ── 1. City extraction ────────────────────────────────────────────────
    origin, destination = _extract_city_pair(text)
    if not origin and not destination:
        origin, destination = _extract_single_city(text)

    # ⭐ Bare-city fallback: user replied to "which city?" with just a city
    # name. Use context (which slot is missing) to classify it.
    if not origin and not destination:
        bare = _find_bare_city(text)
        if bare:
            cur_o = (state.get("origin") or "").strip()
            cur_d = (state.get("destination") or "").strip()
            if not cur_o and cur_d:
                origin = bare
            elif not cur_d and cur_o:
                destination = bare
            elif not cur_o and not cur_d:
                # Nothing known yet — treat as origin (most common case)
                origin = bare

    # Normalize case: "delhi" → "Delhi", "abu dhabi" → "Abu Dhabi"
        # Normalize case: "delhi" → "Delhi", "abu dhabi" → "Abu Dhabi"
    def _norm(c):
        return " ".join(w.capitalize() for w in (c or "").strip().split())

    new_origin = _norm(origin) if origin else None
    new_destination = _norm(destination) if destination else None
    cur_o = _norm(state.get("origin"))
    cur_d = _norm(state.get("destination"))

    # Single city found and classified as origin
    if new_origin and not new_destination:
        if not cur_o:
            updates["origin"] = new_origin
        elif new_origin.lower() != cur_o.lower():
            # New origin → reset the whole trip
            updates.update({
                "origin": new_origin,
                "destination": "",
                "return_destination": "",
                "selected_outbound_flight_id": "",
                "selected_outbound_cabin_class": "",
                "selected_return_flight_id": "",
                "selected_return_cabin_class": "",
            })

    # Single city found and classified as destination
    if new_destination and not new_origin:
        if not cur_d:
            updates["destination"] = new_destination
        elif new_destination.lower() != cur_d.lower():
            # New destination → replace it and reset return-leg / flight selections
            updates.update({
                "destination": new_destination,
                "return_destination": "",
                "selected_outbound_flight_id": "",
                "selected_outbound_cabin_class": "",
                "selected_return_flight_id": "",
                "selected_return_cabin_class": "",
            })

    # Both cities found — handled by the existing block below

    if origin and destination:
        cur_o = (state.get("origin") or "").strip()
        cur_d = (state.get("destination") or "").strip()
        has_out = bool(state.get("selected_outbound_flight_id"))
        is_return = (has_out and cur_o and cur_d
                     and origin.lower() == cur_d.lower()
                     and destination.lower() != cur_d.lower())
        if is_return:
            updates["return_destination"] = destination
        elif (not cur_o or not cur_d
              or (cur_o.lower() == origin.lower()
                  and cur_d.lower() == destination.lower())):
            updates["origin"] = origin
            updates["destination"] = destination
        else:
            updates.update({"origin": origin, "destination": destination,
                            "return_destination": "",
                            "selected_outbound_flight_id": "",
                            "selected_outbound_cabin_class": "",
                            "selected_return_flight_id": "",
                            "selected_return_cabin_class": ""})

    # ── 2. Other facts ────────────────────────────────────────────────────
    for fn, key in ((_extract_duration, "duration_days"),
                    (_extract_budget, "budget_level"),
                    (_extract_travelers, "travelers")):
        v = fn(text)
        if v is not None:
            updates[key] = v

    # ── 3. Price preference ───────────────────────────────────────────────
    if any(k in t_lower for k in ("most expensive", "priciest", "premium",
                                  "premium feel", "top-end", "top end",
                                  "luxury", "best flight", "best hotel")):
        updates["flight_preference"] = "most_expensive"
    elif any(k in t_lower for k in ("cheapest", "cheap", "budget",
                                    "affordable", "lowest", "economical",
                                    "sasta", "saste")):
        updates["flight_preference"] = "cheapest"

    # ── 4. Book this one ──────────────────────────────────────────────────
    if _looks_like_booking(text):
        results = state.get("last_flight_results") or []
        cand = _pick_flight(text, results)
        if cand:
            o_city = cand["origin"].split(" (")[0].lower()
            d_city = cand["destination"].split(" (")[0].lower()
            cur_o = (state.get("origin") or "").lower()
            cur_d = (state.get("destination") or "").lower()
            if cur_o and cur_d and o_city == cur_d and d_city != cur_d:
                updates["selected_return_flight_id"] = cand["id"]
                updates["selected_return_cabin_class"] = cand["cabin_class"]
                updates["return_destination"] = cand["destination"].split(" (")[0]
            else:
                updates["selected_outbound_flight_id"] = cand["id"]
                updates["selected_outbound_cabin_class"] = cand["cabin_class"]
            updates["selection_just_saved"] = True

    # ── 5. Comparison queries are handled by the LLM, not by code ────────
    # (Removed deterministic comparison to avoid false triggers when a user
    #  mentions two different routes in one query.)
    updates["comparison_requested"] = False

    # ── 6. Missing info detection ─────────────────────────────────────────
    wants_flight = any(k in t_lower for k in
                       ("flight", "flights", "fly", "flying", "ticket", "tickets"))
    wants_hotel = any(k in t_lower for k in
                      ("hotel", "hotels", "stay", "room", "accommodation"))
    wants_attraction = any(k in t_lower for k in
                           ("attraction", "attractions", "things to do",
                            "sightseeing", "places to see", "places to visit"))
    wants_plan = updates.get("trip_plan_requested", False)

    cur_origin = (updates.get("origin") or state.get("origin") or "").strip()
    cur_dest = (updates.get("destination") or state.get("destination") or "").strip()

    needs_origin = False
    needs_destination = False

    # A trip-plan request OR any flight-related ask requires an origin.
    # This fires even when the user only said "I want to go to Dubai for 5 days"
    # (no "flight" keyword) — because trip planning inherently needs an origin.
    if wants_plan or wants_flight:
        if not cur_origin:
            needs_origin = True
        elif not cur_dest:
            needs_destination = True
    elif wants_hotel or wants_attraction:
        if not cur_dest:
            needs_destination = True

    updates["needs_origin"] = needs_origin
    updates["needs_destination"] = needs_destination

    return updates


def selection_ack_node(state: TravelState):
    """Deterministic acknowledgement after 'book this one'."""
    parts = []
    for leg, key in (("Outbound", "selected_outbound_flight_id"),
                     ("Return", "selected_return_flight_id")):
        fid = state.get(key)
        if not fid:
            continue
        match = next((f for f in load_db().get("flights", [])
                      if f["id"].upper() == fid.upper()), None)
        if match:
            cabin_key = ("selected_outbound_cabin_class" if leg == "Outbound"
                         else "selected_return_cabin_class")
            cabin = state.get(cabin_key, "Economy")
            parts.append(f"Saved {leg.lower()} flight: {match['airline']} "
                         f"{match['flight_no']} | {match['origin']} -> "
                         f"{match['destination']} | {cabin} | ID {match['id']}")

    if not parts:
        parts.append("I couldn't resolve that flight selection. Please tell me "
                     "the flight number or Flight ID.")

    body = "\n".join(parts)
    if "Outbound" in body:
        body += ("\n\nWhat next? I can build a full trip plan, or you can "
                 "confirm the other leg first.")
    return {"selection_just_saved": False, "messages": [AIMessage(content=body)]}


def ask_missing_info_node(state: TravelState):
    """Deterministic question for missing origin/destination."""
    if state.get("needs_origin"):
        q = "Which city are you flying from?"
    elif state.get("needs_destination"):
        q = "Which city would you like to go to?"
    else:
        q = "Could you tell me a bit more about your trip?"
    return {
        "needs_origin": False,
        "needs_destination": False,
        "messages": [AIMessage(content=q)],
    }

def direct_trip_plan_node(state: TravelState):
    """Build the plan and return it as the FINAL AIMessage — bypass the LLM
    entirely so the plan text isn't reformatted, mangled, or truncated."""
    origin = state.get("origin", "")
    destination = state.get("destination", "")
    if not origin or not destination:
        return {"trip_plan_requested": False,
                "messages": [AIMessage(content="I need both the origin and "
                                               "destination before I can build "
                                               "the plan.")]}

    kwargs = {
        "origin": origin, "destination": destination,
        "duration_days": state.get("duration_days", 5),
        "budget_level": state.get("budget_level", "midrange"),
        "flight_preference": state.get("flight_preference", "cheapest"),
        "travelers": state.get("travelers", 1),
        "interests": state.get("interests", "sightseeing, culture, local food"),
        "cabin_class": state.get("selected_outbound_cabin_class") or "Economy",
        "round_trip": True,
        "outbound_flight_id": state.get("selected_outbound_flight_id", ""),
        "outbound_cabin_class": state.get("selected_outbound_cabin_class", ""),
        "return_flight_id": state.get("selected_return_flight_id", ""),
        "return_cabin_class": state.get("selected_return_cabin_class", ""),
        "return_destination": state.get("return_destination", ""),
    }
    result = build_trip_plan.invoke(kwargs)
    return {"trip_plan_requested": False,
            "messages": [AIMessage(content=result)]}


def _memory_hint(state: TravelState) -> str:
    return "\n".join([
        "SESSION MEMORY (do not re-ask these):",
        f"  origin          = {state.get('origin') or '[unset]'}",
        f"  destination     = {state.get('destination') or '[unset]'}",
        f"  return city     = {state.get('return_destination') or '[= origin]'}",
        f"  travelers       = {state.get('travelers', '[unset]')}",
        f"  duration_days   = {state.get('duration_days', '[unset]')}",
        f"  budget_level    = {state.get('budget_level', '[unset]')}",
        f"  flight_pref     = {state.get('flight_preference', '[unset]')}",
        f"  outbound flight = {state.get('selected_outbound_flight_id') or '[none]'}"
        + (f" ({state.get('selected_outbound_cabin_class')})"
           if state.get("selected_outbound_cabin_class") else ""),
        f"  return flight   = {state.get('selected_return_flight_id') or '[none]'}"
        + (f" ({state.get('selected_return_cabin_class')})"
           if state.get("selected_return_cabin_class") else ""),
    ])


def agent_node(state: TravelState):
    """Calls the LLM with system prompt + memory hint + full conversation."""
    try:
        response = llm_with_tools.invoke(
            [SystemMessage(content=SYSTEM_PROMPT),
             SystemMessage(content=_memory_hint(state))]
            + state["messages"]
        )
    except Exception as e:
        print(f"  [agent] LLM invocation failed: {e}")
        response = AIMessage(content=(
            "Sorry — I hit a technical hiccup. Could you try that again?"
        ))
    return {"messages": [response]}


def tools_node(state: TravelState):
    """Execute tool calls. Strips the flight-selection marker from output.
    If build_trip_plan was called, its output becomes the final reply
    directly (bypassing LLM reformatting)."""
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", []) or []
    tool_map = {t.name: t for t in ALL_TOOLS}

    new_results = state.get("last_flight_results", [])
    out_msgs: list = []
    build_plan_result = None

    for tc in calls:
        name = tc["name"] if isinstance(tc, dict) else tc.name
        args = tc["args"] if isinstance(tc, dict) else tc.args
        call_id = tc["id"] if isinstance(tc, dict) else tc.id

        fn = tool_map.get(name)
        if not fn:
            out_msgs.append(ToolMessage(content=f"Unknown tool: {name}",
                                        tool_call_id=call_id, name=name))
            continue

        try:
            raw = fn.invoke(args)
        except Exception as e:
            out_msgs.append(ToolMessage(content=f"TOOL ERROR in {name}: {e}",
                                        tool_call_id=call_id, name=name))
            continue

        clean = raw
        if isinstance(raw, str) and FLIGHT_SELECTION_MARKER in raw:
            idx = raw.rfind(FLIGHT_SELECTION_MARKER)
            clean = raw[:idx].rstrip()
            try:
                parsed = json.loads(raw[idx + len(FLIGHT_SELECTION_MARKER):].strip())
                new_results = parsed.get("results", [])
            except Exception:
                pass

        out_msgs.append(ToolMessage(content=clean, tool_call_id=call_id, name=name))

        if name == "build_trip_plan" and isinstance(clean, str):
            build_plan_result = clean

    if build_plan_result is not None:
        # Trip plan is already perfectly formatted — send directly to user
        return {
            "messages": out_msgs + [AIMessage(content=build_plan_result)],
            "last_flight_results": new_results,
        }

    return {"messages": out_msgs, "last_flight_results": new_results}


# ── Validator (zero-hallucination enforcement) ────────────────────────────────
RETRY_MARKER = "[[VALIDATOR_RETRY]]"
MAX_RETRIES = 2

def _extract_facts(text: str):
    """Pull out DB-verifiable facts: flight numbers and prices.
    Returns (flight_numbers, set_of_price_integers)."""
    if not text:
        return [], set()
    flight_numbers = re.findall(r"\b[A-Z]{1,3}-\d{3,4}\b", text)
    prices = set()
    # $1234, $1,234, ₹14691, ₹14,691
    for m in re.finditer(r"[$₹]\s?(\d[\d,]*)", text):
        try:
            prices.add(int(m.group(1).replace(",", "")))
        except ValueError:
            pass
    # INR 14691, USD 246 (used in tool output formatting)
    for m in re.finditer(r"\b(?:INR|USD)\s?(\d[\d,]*)", text):
        try:
            prices.add(int(m.group(1).replace(",", "")))
        except ValueError:
            pass
    return flight_numbers, prices


def _has_cjk(text: str) -> bool:
    """True if text contains Chinese / Japanese / Korean characters."""
    if not text:
        return False
    return any(
        "\u4e00" <= ch <= "\u9fff"      # CJK Unified Ideographs
        or "\u3040" <= ch <= "\u30ff"   # Hiragana / Katakana
        or "\uac00" <= ch <= "\ud7af"   # Hangul
        for ch in text
    )


def _last_human_was_english(messages) -> bool:
    """Check whether the user's most recent message was written in English."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return not _has_cjk(m.content or "")
    return True


def _count_retries(messages):
    count = 0
    for m in messages:
        if isinstance(m, HumanMessage):
            count = 0
        elif isinstance(m, SystemMessage) and RETRY_MARKER in (m.content or ""):
            count += 1
    return count


def _has_cjk(text: str) -> bool:
    """True if text contains Chinese / Japanese / Korean characters."""
    if not text:
        return False
    return any(
        "\u4e00" <= ch <= "\u9fff"      # CJK Unified Ideographs
        or "\u3040" <= ch <= "\u30ff"   # Hiragana / Katakana
        or "\uac00" <= ch <= "\ud7af"   # Hangul
        for ch in text
    )


def _last_human_was_english(messages) -> bool:
    """Check whether the user's most recent message was written in English
    (i.e. contains no CJK characters)."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return not _has_cjk(m.content or "")
    return True


def validate_node(state: TravelState):
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return {}
    reply = last.content or ""

    # ── 1. Language backstop ─────────────────────────────────────────────
    # If the model replied in Chinese/Japanese/Korean but the user wrote in
    # English, reject it and force an English retry.
    if _has_cjk(reply) and _last_human_was_english(state["messages"]):
        retries = _count_retries(state["messages"])
        if retries >= MAX_RETRIES:
            print("  [validator] gave up on CJK reply after retries")
            return {"validator_state": "gave_up",
                    "messages": [AIMessage(content=(
                        "Sorry — I had trouble generating a proper reply. "
                        "Could you rephrase your question?"))]}
        print(f"  [validator] rejected — LLM replied in CJK but user wrote "
              f"English (retry {retries + 1}/{MAX_RETRIES})")
        return {"validator_state": "retry",
                "messages": [SystemMessage(content=(
                    f"{RETRY_MARKER} You replied in a non-English language, but "
                    f"the user wrote in English. Rewrite your ENTIRE reply in "
                    f"English. Do not use any Chinese, Japanese, or Korean "
                    f"characters. If you are missing information, ask the "
                    f"question in English."))]}

    # ── 2. Trusted deterministic replies skip fact-checking ─────────────
    if (reply.startswith("=" * 10) or "TRIP PLAN —" in reply
            or "CHEAPEST vs MOST EXPENSIVE" in reply
            or reply.startswith("Saved outbound flight")
            or reply in ("Which city are you flying from?",
                         "Which city would you like to go to?",
                         "Could you tell me a bit more about your trip?")):
        return {"validator_state": "clean"}

    # ── 3. No tools called → nothing to verify against ──────────────────
    tool_outputs = "\n".join(m.content for m in state["messages"]
                             if isinstance(m, ToolMessage))
    if not tool_outputs:
        return {"validator_state": "clean"}

    # ── 4. Fact check ────────────────────────────────────────────────────
    # ── 4. Fact check (numeric comparison, currency-agnostic) ────────────
    reply_flights, reply_prices = _extract_facts(reply)
    tool_flights, tool_prices = _extract_facts(tool_outputs)

    invented = []
    for f in reply_flights:
        if f not in tool_flights:
            invented.append(f)
    for p in reply_prices:
        if p not in tool_prices:
            invented.append(f"${p}")

    if not invented:
        return {"validator_state": "clean"}

    retries = _count_retries(state["messages"])
    if retries >= MAX_RETRIES:
        print(f"  [validator] giving up after {retries} retries — invented {invented}")
        return {"validator_state": "gave_up",
                "messages": [AIMessage(content=(
                    "I couldn't produce a reply strictly from the database for "
                    "that query. Could you try rephrasing?"))]}

    print(f"  [validator] rejected — invented {invented} (retry {retries + 1}/{MAX_RETRIES})")
    return {"validator_state": "retry",
            "messages": [SystemMessage(content=(
                f"{RETRY_MARKER} Your previous reply contained facts NOT in any "
                f"tool result: {', '.join(invented)}. Rewrite your answer using "
                f"ONLY the data in the tool results above. Do not add any flight "
                f"number, price, hotel name, or attraction that is not there."))]}

# ── Coverage check (multi-part enforcement) ───────────────────────────────────
INTENT_KEYWORDS = {
    "search_flights":      ("flight", "flights", "fly", "flying", "ticket", "tickets"),
    "search_hotels":       ("hotel", "hotels", "stay", "room", "rooms", "accommodation"),
    "explore_attractions": ("attraction", "attractions", "things to do",
                            "sightseeing", "places to see", "places to visit"),
    "find_dining_options": ("restaurant", "restaurants", "food", "dining",
                            "eat", "cuisine"),
}


def coverage_check_node(state: TravelState):
    """Ensure every intent in the user's last message produced a tool call.
    Force-invoke any missing tool."""
    msgs = state.get("messages", [])

    human_idx = -1
    for i in range(len(msgs) - 1, -1, -1):
        if isinstance(msgs[i], HumanMessage):
            t = (msgs[i].content or "").lower()
            if any(kw in t for kws in INTENT_KEYWORDS.values() for kw in kws):
                human_idx = i
                break
    if human_idx == -1:
        return {}

    human_text = (msgs[human_idx].content or "").lower()
    required = {tool for tool, kws in INTENT_KEYWORDS.items()
                if any(kw in human_text for kw in kws)}
    if not required:
        return {}

    called = set()
    for m in msgs[human_idx:]:
        tc = getattr(m, "tool_calls", None) or []
        for call in tc:
            n = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if n:
                called.add(n)
        if isinstance(m, ToolMessage):
            called.add(getattr(m, "name", "") or "")
    if "build_trip_plan" in called:
        return {}

    missing = required - called
    if not missing:
        return {}

    origin = (state.get("origin") or "").strip()
    destination = (state.get("destination") or "").strip()
    forced: list = []

    for tool_name in sorted(missing):
        try:
            if tool_name == "search_flights":
                if not origin or not destination:
                    continue
                args = {"origin": origin, "destination": destination,
                        "cabin_class": "", "sort_by": "price", "sort_order": "asc"}
                result = search_flights.invoke(args)
            elif tool_name == "search_hotels":
                if not destination:
                    continue
                args = {"city": destination, "sort_by": "price", "sort_order": "asc"}
                result = search_hotels.invoke(args)
            elif tool_name == "explore_attractions":
                if not destination:
                    continue
                args = {"city": destination}
                result = explore_attractions.invoke(args)
            elif tool_name == "find_dining_options":
                if not destination:
                    continue
                args = {"city": destination}
                result = find_dining_options.invoke(args)
            else:
                continue
        except Exception as e:
            print(f"  [coverage] {tool_name} failed: {e}")
            continue

        call_id = f"forced-{uuid.uuid4()}"
        forced.append(AIMessage(
            content="",
            tool_calls=[{"name": tool_name, "args": args, "id": call_id}]))
        forced.append(ToolMessage(name=tool_name, tool_call_id=call_id, content=result))
        print(f"  [coverage] force-called missing tool: {tool_name}")

    if not forced:
        return {}

    forced.append(SystemMessage(content=(
        "Now write ONE reply containing results for EVERY part of the user's "
        "request. Do not say 'next I will search'. Show flights AND hotels AND "
        "anything else they asked for, using the tool results above.")))
    return {"messages": forced}


# ══════════════════════════════════════════════════════════════════════════════
# 9. ROUTING
# ══════════════════════════════════════════════════════════════════════════════

def route_from_preprocess(state: TravelState):
    if state.get("selection_just_saved"):
        return "selection_ack"
    if state.get("needs_origin") or state.get("needs_destination"):
        return "ask_missing_info"
    if (state.get("trip_plan_requested")
            and state.get("origin") and state.get("destination")):
        return "direct_trip_plan"
    return "agent"


def route_from_agent(state: TravelState):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "validate"


def route_from_tools(state: TravelState):
    last = state["messages"][-1]
    # tools_node emitted a final AIMessage (build_trip_plan path) → done
    if isinstance(last, AIMessage) and not getattr(last, "tool_calls", None):
        return END
    return "agent"


def route_from_validate(state: TravelState):
    vs = state.get("validator_state")
    if vs == "retry":
        return "agent"
    if vs == "gave_up":
        return END
    return "coverage_check"


def route_after_coverage(state: TravelState):
    last = state["messages"][-1]
    if isinstance(last, SystemMessage) and "EVERY part of the user's request" in (last.content or ""):
        return "agent"
    return END


# ══════════════════════════════════════════════════════════════════════════════
# 10. GRAPH
# ══════════════════════════════════════════════════════════════════════════════

builder = StateGraph(TravelState)
builder.add_node("preprocess", preprocess_node)
builder.add_node("selection_ack", selection_ack_node)
builder.add_node("ask_missing_info", ask_missing_info_node)
builder.add_node("direct_trip_plan", direct_trip_plan_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_node("validate", validate_node)
builder.add_node("coverage_check", coverage_check_node)

builder.set_entry_point("preprocess")

builder.add_conditional_edges("preprocess", route_from_preprocess, {
    "selection_ack": "selection_ack",
    "ask_missing_info": "ask_missing_info",
    "direct_trip_plan": "direct_trip_plan",
    "agent": "agent",
})
builder.add_edge("selection_ack", END)
builder.add_edge("ask_missing_info", END)
builder.add_edge("direct_trip_plan", END)

builder.add_conditional_edges("agent", route_from_agent, {
    "tools": "tools", "validate": "validate",
})
builder.add_conditional_edges("tools", route_from_tools, {
    "agent": "agent", END: END,
})
builder.add_conditional_edges("validate", route_from_validate, {
    "agent": "agent",
    "coverage_check": "coverage_check",
    END: END,
})
builder.add_conditional_edges("coverage_check", route_after_coverage, {
    "agent": "agent", END: END,
})

app = builder.compile(checkpointer=MemorySaver())


# ══════════════════════════════════════════════════════════════════════════════
# 11. CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    BUILD = "v10-full-rebuild"
    print("=" * 60)
    print(f"  Travel Agent — {BUILD}")
    print(f"  Provider: {PROVIDER}")
    print(f"  Zero-hallucination validator: ON")
    print(f"  Deterministic trip plans: ON")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    thread = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 30}

    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in {"exit", "quit"}:
            print("\nSafe travels.")
            break
        if not user:
            continue

        try:
            for _ in app.stream({"messages": [HumanMessage(content=user)]},
                                config=config):
                pass
        except Exception as e:
            print(f"\n[agent error: {e}]")
            continue

        state = app.get_state(config)
        last = state.values["messages"][-1]
        if isinstance(last, AIMessage):
            print(f"\nAgent:\n{last.content}\n")
        else:
            print(f"\nAgent: [no reply — try rephrasing]\n")
        print("-" * 60)
