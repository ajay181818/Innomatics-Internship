# -*- coding: utf-8 -*-
"""Python If-Else.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rcddYS055QEKVhLDMjs1wBYHBbZ5kBFZ
"""

n = int(input().strip())

if n % 2 != 0:
    print ("Weird")
else:
    if n >= 2 and n <= 5:
        print ("Not Weird")
    elif n >= 6 and n <= 20:
        print ("Weird")
    elif n > 20:
        print ("Not Weird")