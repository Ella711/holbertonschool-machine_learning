#!/usr/bin/env python3
"""
30. List all documents in Python
"""


def list_all(mongo_collection):
    """
    Lists all documents in a collection
    Args:
        mongo_collection: pymongo collection object

    Returns: empty list if no document in the collection
    """
    return list(mongo_collection.find())
