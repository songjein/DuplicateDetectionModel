import random

wf1 = open("ts1.txt", "a")
wf2 = open("ts2.txt", "a")
wf3 = open("ts3.txt", "a")


def getRandList1(l):
	return [str(random.randrange(1,20)) for i in range(l)]

def getRandList2(l):
	return [str(random.randrange(120,200)) for i in range(l)]

for i in range(1000):
	wf1.write(",".join(getRandList1(10)) + "\n")
	wf2.write(",".join(getRandList1(10)) + "\n")
	wf3.write("%s\n" %(1))

for i in range(1000):
	wf1.write(",".join(getRandList1(10)) + "\n")
	wf2.write(",".join(getRandList2(10)) + "\n")
	wf3.write("%s\n" %(0))
