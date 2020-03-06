import re
s = "I am human." * 100000000
#print(len(re.findall('human', s)))
print(s.count('human'))
