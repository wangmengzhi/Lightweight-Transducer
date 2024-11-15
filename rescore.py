from train import wer
import timeit
import kenlm
import jieba
import os

if not os.path.exists('dict.txt'):
    f_dict = open('dict.txt', 'w', encoding='utf-8')
    with open('data/aishell/resource_aishell/lexicon.txt', encoding='utf-8') as f:
        for l in f:
            f_dict.write(l.split()[0]+'\n')
    f_dict.close()

jieba.load_userdict('dict.txt')

model = kenlm.Model('aishell_train.arpa')

lm_weight=0.4
total_d=0
total_l=0
total_sub=0
total_ins=0
total_dele=0
cnt=0
recs=[]
scores=[]
start_time = timeit.default_timer()
with open('nbest.txt', encoding='utf-8') as f:
    for l in f:
        l=l.strip().split(',')
        if len(l)==1:
            max_score=-float('inf')
            for rec, score in zip(recs, scores):
                r=' '.join(jieba.cut(rec.replace(' ',''), HMM=False))
                lm_score=model.score(r, bos = False, eos = False)
                score=score+lm_score*lm_weight
                if score>max_score:
                    max_score=score
                    best=rec
            (d,ins,dele,sub),l = wer(l[0], best)
            total_d+=d
            total_l+=l            
            total_sub+=sub
            total_ins+=ins
            total_dele+=dele
            cnt+=1
            if cnt%1000==0:print(cnt)
            recs.clear()
            scores.clear()
        else:
            rec,score=l
            recs.append(rec)
            scores.append(float(score))
print('wer:%.2f S=%.2f I=%.2f D=%.2f'%(total_d/total_l*100,total_sub/total_l*100,total_ins/total_l*100,total_dele/total_l*100))
time=timeit.default_timer()-start_time
print('time:',int(time%3600//60),'m',int(time%60),'s')