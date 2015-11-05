from sklearn import svm
import json
import numpy as np
import random



data = []
with open('final.txt') as f:
    for line in f:
    	data.append(line)
    count =1
    di = {}	
    for i in data:
    	str_ = i
	l = str_.split( )
	for j in l:
	   if not di.has_key(j):
	      di[j] =count
	      count = count +1 

data1 = []
answer = []
inputarray = []
outputarray = []
count = 0
with open('final.txt') as f:
    for line in f:
        data1 = []
	answer = []
	answer1 = []
	#inputarray = []
	#outputarray = []
	count = 0
	data1 = line.split()
	#print data
	for i in data1:
	    #print count,
	    if len(data1) > 0:
		if count < (len(data1)-2):
 		    answer = []
	            answer1 = []
		    answer.append(di[data1[count]])
		    outputarray.append(di[data1[count+1]])
		   # answer1.append(di[data1[count+1]])
		    answer.append(di[data1[count+2]])
		    inputarray.append(answer)
		   # outputarray.append(answer1)
		    count = count+1
    #if len(inputarray) > 0:
     #   print inputarray
     #   print ""
   # if len(outputarray) > 0:
    #    print outputarray
    #    print ""
		
out =[]
out.append(outputarray)
#sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array(inputarray)
    
 # output dataset            
y = np.array(out).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((2,50)) - 1
#print syn0

for iter in xrange(1):

    # forward propagation
	l0 = X
	l1 = nonlin(np.dot(l0,syn0))
	#print l1

    # how much did we miss?
	l1_error = y - l1
	#print y
	#print l1_error

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
	l1_delta = l1_error * nonlin(l1,True)

    # update weights
	syn0 += np.dot(l0.T,l1_delta)

#print "Output After Training:"
#print l1[0]
#print len(l1)
#print len(inputarray)

#classifier
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
		   index= out[0].index(num)
		   vec = [x+y for x,y in zip(vec, l1[index])]
		except Exception,e:
		   cout = 0
		   for j in inputarray:
		      try:
			 h = j.index(num)
			 index = cout
                         break
		      except Exception,e:
                         pass
                      cout = cout +1 
		   vec = [x+y for x,y in zip(vec, l1[index])]	
		   				
            else:
		vec = [random.random()]*50
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
		   cout = 0
		   for j in inputarray:
		      try:
			 h = j.index(num)
			 index = cout
                         break
		      except Exception,e:
                         pass
                      cout = cout +1 
		   vec = [x+y for x,y in zip(vec, l1[index])]	
		   				
            else:
		vec = [random.random()]*50
        #print vec
	t2.append(vec)
        count = count +1  		  
clf = svm.SVC()
clf.fit(t1_x, t1_y)


final = clf.predict(t2)

print t1_y
for i in final:
   print i


