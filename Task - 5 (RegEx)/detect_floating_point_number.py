# -*- coding: utf-8 -*-
"""Detect Floating Point Number.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Bq5Cej5n8IJQPaJ4j0QIY2c5Y9z5P_xF
"""



from re import match, compile
pattern = compile('^[-+]?[0-9]*\.[0-9]+$')
for  in range(int(input())):
    print(bool(pattern.match(input())))