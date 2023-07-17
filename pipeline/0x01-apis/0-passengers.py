#!/usr/bin/env python3
"""
0. Can I join?
"""
import requests


def availableShips(passengerCount):
    """
    Returns the list of ships that can hold a given number of passengers

    Args:
        passangerCount: number of passengers to look for

    Return: List of ships or empty list
    """
    url = "https://swapi-api.hbtn.io/api/starships/"
    resp_ships = requests.get(url)
    json = resp_ships.json()
    results = json["results"]
    ships = []
    while json["next"]:
        for ship in results:
            if ship["passengers"] == 'n/a' or ship["passengers"] == 'unknown':
                continue
            if int(ship["passengers"].replace(',', '')) >= passengerCount:
                ships.append(ship["name"])
        next_url = json["next"]
        resp_ships = requests.get(next_url)
        json = resp_ships.json()
        results = json["results"]
    return ships
