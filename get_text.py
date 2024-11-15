import csv

data={}
f=open('data/aishell/aishell_transcript_v0.8.txt',encoding='utf-8')
for i in f.readlines():
    l=i.split()
    fn=l[0]
    text=' '.join(l[1:])
    data[fn]=text
    
f_out=open('aishell_train.txt','w',encoding='utf-8')
fs=['data/aishell/aishell_train.csv']
for f in fs:
    r=csv.reader(open(f,encoding='utf-8'))
    for i,l in enumerate(r):
        if i==0:continue
        if i%10000==0:print(i)
        name=l[0].split('/')[-1].split('.')[0]
        f_out.write(data[name]+'\n')
f_out.close()
