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
simple=[]
final_join1 = []
count_50 = 0
with open("vector1.txt") as f1:
    for line in f1:
##print line 
        joining1 = line.split(" ")
        for i in joining1 :
            simple.append(i)
            count_50 = count_50 + 1
            if count_50 == 50:
                final_join1.append(simple)
                count_50 = 0
                simple=[]
                
#print final_join1
a=0.5
b=0.5
y=len(final_join)
z=len(final_join1)
print y
print z
ans=[]
final =[]
for i in range(0,y):
    ans=[]
    for j in range(0,50):
#print j
	c = float(final_join[i][j]) + float(final_join1[i][j])
	ans.append(c)
    final.append(ans)	
print final

    
     
