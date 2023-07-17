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
    getreq_starship = requests.get(url)
    ship_list = []
    while getreq_starship.status_code == 200:
        for ship in getreq_starship.json()['results']:
            if ship["passengers"] is not None:
                try:
                    num_pasgrs = ship["passengers"].replace(",", "")
                    if int(num_pasgrs) >= passengerCount:
                        ship_list.append(ship["name"])
                except ValueError:
                    pass
        try:
            next = requests.get(getreq_starship.json()["next"])
        except Exception:
            break
    return ship_list
