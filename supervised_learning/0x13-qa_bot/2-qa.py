#!/usr/bin/env python3
"""
2. Answer Questions
"""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Answers questions from a reference text

    Args:
        reference: reference text

    Returns: an Answer
    """
    exits = ('exit', 'quit', 'goodbye', 'bye')

    while (True):
        user_Q = input("Q: ")
        if user_Q.lower() in exits:
            print('A: Goodbye')
            break
        else:
            answer = question_answer(user_Q, reference)
            if answer:
                print('A: ', answer)
            else:
                print('A: Sorry, I do not understand your question.')
