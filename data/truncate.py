f1 = open("truncate.txt", 'w', encoding='UTF8')
f2 = open("glove.840B.300d.txt", 'r', encoding='UTF8')
i=0
while i < 10000:
    line = f2.readline()
    i+=1
    f1.write(line)
f2.close()
f1.close()