from sklearn import svm
import json
import numpy as np
import random



data = []
with open('sswe_token.txt') as f:
    for line in f:
    	data.append(line)
    count =1
    di = {}
    lin =0	
    for i in data:
    	str_ = i
	lin = lin +1
	l = str_.split( )
	for j in l:
	   if not di.has_key(j):
	      di[j] =count
	      print j,count		
	      count = count +1

