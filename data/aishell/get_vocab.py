import csv
    
d=[]
fs=['data/aishell/aishell_train.csv']
for f in fs:
    r=csv.reader(open(f,encoding='utf-8'))
    for i,l in enumerate(r):
        if i==0:continue
        if i%10000==0:print(i)
        for c in l[2]:
            if c!=' ' and c not in d:
                d.append(c)

d=sorted(d)
d=['<s>','</s>','<unk>']+d
print(len(d))
with open('data/aishell/vocab.txt','w',encoding='utf-8') as f:
    for c in d:
        f.write(c+'\n')
