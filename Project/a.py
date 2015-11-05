import json

data = []
with open('SSWE.txt') as f:
    for line in f:
	    j = line.split('":')
	    a = j[3].replace(',"label','')
            data.append(a)
    
    for i in data:        
        j = i.replace('"','')
	print j
	#label.append(j[1])
    #print label
		
#temp = ''.join([i[u'text']])
#   	    answer.append(temp)
#   print answer
#print i[u'text']print"\t"
#	    print i[u'_id']
#	    print ""

