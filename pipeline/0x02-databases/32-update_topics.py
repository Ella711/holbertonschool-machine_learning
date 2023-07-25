#!/usr/bin/env python3
"""
32. Change school topics
"""


def update_topics(mongo_collection, name, topics):
    """
    Changes all topics of a school document based on the name
    Args:
        mongo_collection: pymongo collection object
        name (string): school name to update
        topics: list of topics approached in the school
    """
    search = {"name": name}
    new = {"$set": {"topics": topics}}
    mongo_collection.update_many(search, new)
