import json

data = []
with open('s') as f:
    count =1
    for line in f:
	    j = line.split('":')
            l = len(j)	
	    a = j[4].replace('}','')
	   #print count,a
            count =count +1	
            data.append(a)
    
    	
    for i in data:        
	print i
	#label.append(j[1])
    #print label
		
#temp = ''.join([i[u'text']])
#   	    answer.append(temp)
#   print answer
#print i[u'text']print"\t"
#	    print i[u'_id']
#	    print ""

