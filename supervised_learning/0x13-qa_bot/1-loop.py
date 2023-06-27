#!/usr/bin/env python3
"""
1. Create the loop
"""

exits = ('exit', 'quit', 'goodbye', 'bye')

while (True):
    user_Q = input("Q: ")
    if user_Q.lower() in exits:
        print('A: Goodbye')
        break
    else:
        print('A: ')
