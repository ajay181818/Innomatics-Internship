# -*- coding: utf-8 -*-
"""Mutations.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TZa_H5ZsZgY-czIJZrit-XPL46IeUla7
"""

def mutate_string(string, position, character):
    n = list(string)
    n[position] = character
    string = "".join(n)
    return string
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)