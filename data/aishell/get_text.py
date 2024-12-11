import csv
    
f_out=open('data/aishell/aishell_train.txt','w',encoding='utf-8')
fs=['data/aishell/aishell_train.csv']
for f in fs:
    r=csv.reader(open(f,encoding='utf-8'))
    for i,l in enumerate(r):
        if i==0:continue
        if i%10000==0:print(i)
        f_out.write(l[2]+'\n')
f_out.close()
