
from sklearn import svm
import json
import numpy as np
import random

data = []
word_count =0
with open('dict.txt') as f:
    for line in f:
    	data.append(line)
    count =1
    di = {}	
    for i in data:
    	str_ = i
	l = str_.split( )
	di[l[0]]=l[1]
        word_count = word_count +1

 

#print di


final_join = []
simple = []
count_50 = 0
with open("vector.txt") as f:
    for line in f:
##print line
	joining = line.split(" ")
	for i in joining :
	    simple.append(i)
	    count_50 = count_50 + 1
	    if count_50 == 50:
	        final_join.append(simple)
		count_50 = 0
		simple=[]

#print final_join

l1 = final_join
t1_x =[]
t1_y  =[]
 
t2 =[]

data = []
with open('train.tsv') as f:
    for line in f:
            data.append(line)
    
    label = []
    for i in data:        
        j = i.split("\t")
	label.append(j[2].replace("\n",""))
   # print label

#print len(label)
train_label = label

count = 0
vec = [0] * 50
with open('train_token.txt') as f:
    for line in f:
	words = len(line)
	vec = [0] * 50
	for i in line:
            if di.has_key(i):
		num = di[i]
		try:
		   index= num
		   vec = [x+y for x,y in zip(vec, l1[index])]
		except Exception,e:
		   ran =[]
		   for r in range(0,50):
                       b = random.uniform(-1,1)
                       ran.append(b) 	
		   vec = [x+y for x,y in zip(vec, ran)]	
		   				
            else:
		ran =[]
	        for r in range(0,50):
                    b = random.uniform(-1,1)
                    ran.append(b)
                vec =ran	  
         vec = vec/words         
       # print vec,",",label[count]
	t1_x.append(vec)
        count = count +1     
    #print count
t1_y = label

data = []
with open('test.tsv') as f:
    for line in f:
            data.append(line)
    
    label = []
    for i in data:        
        j = i.split("\t")
	label.append(j[2].replace("\n",""))
   # print label

#print len(label)

count = 0
vec = [0] * 50
with open('test_token.txt') as f:
    for line in f:
	words = len(line)
	vec = [0] * 50
	for i in line:
            if di.has_key(i):
		num = di[i]
		try:
		   index= out[0].index(num)
		   vec = [x+y for x,y in zip(vec, l1[index])]
		except Exception,e:
		   ran =[]
		   for r in range(0,50):
                       b = random.uniform(-1,1)
                       ran.append(b) 	
		   vec = [x+y for x,y in zip(vec, ran)]	
		   				
            else:
		ran =[]
	        for r in range(0,50):
                    b = random.uniform(-1,1)
                    ran.append(b)
                vec =ran
        #print vec
	vec = vec/words
	t2.append(vec)
        count = count +1  

		  
clf = svm.SVC()
clf.fit(t1_x, t1_y)


final = clf.predict(t2)

print t1_y
for i in final:
   print i

