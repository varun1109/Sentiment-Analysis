import json

data = []
answer = []
inputarray = []
outputarray = []
count = 0
with open('tweet.txt') as f:
    for line in f:
		data = []
		answer = []
		answer1 = []
		#inputarray = []
		#outputarray = []
		count = 0
		data = line.split()

		#print data
		for i in data:
			#print count,
			if len(data) > 0:
				if count < (len(data)-2):
					answer = []
					answer1 = []
					answer.append(data[count])
					answer1.append(data[count+1])
					answer.append(data[count+2])
					inputarray.append(answer)
					outputarray.append(answer1)
					count = count+1
		if len(inputarray) > 0:
			print inputarray
		print ""
		if len(outputarray) > 0:
			print outputarray
		print ""


			#for j in i:
				#print j 
					    
    	#for i in data:
    	 #   print i
    	  #  print i+1
    	   # print i+2
	    #print ""
#temp = ''.join([i[u'text']])
#   	    answer.append(temp)
#   print answer
#print i[u'text']print"\t"
#	    print i[u'_id']
#	    print ""

