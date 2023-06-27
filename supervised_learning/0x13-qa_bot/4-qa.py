#!/usr/bin/env python3
"""
4. Multi-reference Question Answering
"""
singular_question_answer = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """
    Answers questions from multiple reference texts

    Args:
        corpus_path: path to the corpus of reference documents
    """
    exits = ('exit', 'quit', 'goodbye', 'bye')

    while (True):
        user_Q = input("Q: ")
        if user_Q.lower() in exits:
            print('A: Goodbye')
            break
        else:
            reference = semantic_search(corpus_path, user_Q)
            answer = singular_question_answer(user_Q, reference)
            if answer:
                print('A: ', answer)
            else:
                print('A: Sorry, I do not understand your question.')
