#!/usr/bin/env python3
"""
1. Where I am?
"""
import requests


def sentientPlanets():
    """
    Returns the list of names of the home planets of all sentient species
    """
    url = "https://swapi-api.hbtn.io/api/species"
    getreq_species = requests.get(url)
    planet_list = []
    while getreq_species.status_code == 200:
        for species in getreq_species.json()["results"]:
            url = species["homeworld"]
            if url is not None:
                homeworld = requests.get(url)
                planet_list.append(homeworld.json()["name"])
        try:
            getreq_species = requests.get(getreq_species.json()["next"])
        except Exception:
            break
    return planet_list
