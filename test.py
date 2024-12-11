import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from train import Model,get_audio,i2s,i2s_ctc,get_vocab,SpeechDataset,collate_fn,wer
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timeit

w2i,i2w=get_vocab()

model = Model(256)
print(f'Total parameters: {sum(p.numel() for p in model.parameters())/1024/1024:.2f}M')

model.load_state_dict(torch.load('save/model.avg'))
device = torch.device("cuda:0")
model.to(device)
model.eval()

beam=10
sos_id=w2i['<s>']
eos_id=w2i['</s>']
blank_id=0
#nbest_path=None
nbest_path='nbest.txt'
if nbest_path:
    f=open(nbest_path, 'w', encoding='utf-8')
    
def infer(x):
    if isinstance(x, str):
        x=get_audio(x,False)
        x=x.unsqueeze(0).to(device)
    x=model.get_feat(x)
    x=model.encoder(x)
    T,_,H=x.shape
    
    last_pred=x.new_ones(beam,dtype=torch.long)*sos_id
    emb=model.emb(last_pred.long())
    lm_state=F.pad(emb.unsqueeze(-1), [model.context_size-1,0])
    lm=model.lm(lm_state).squeeze(-1)
    lm_logit=model.lm_out(lm)
    am_logit=model.am_out(x)
    nbest_score=x.new_zeros(beam)
    nbest=x.new_zeros(0,beam).long()

    enc=x.repeat(1,beam,1)
    last_ct=x.new_zeros(beam,H)
    for k in range(T):
        blank=torch.cat([enc[k],emb,last_ct],dim=-1)
        blank=model.blank(blank)
        logit=am_logit[k]+lm_logit
        blank2=F.logsigmoid(blank)
        logit=logit[:,1:].log_softmax(-1)+blank2-blank
        logit=torch.cat([blank2,logit],dim=-1)
        logit*=0.8
        logit=logit.log_softmax(-1)
        V=logit.shape[-1]
        nbest_score=nbest_score.reshape(beam,1)+logit 
        if nbest.shape[0]==0:
            nbest_score=nbest_score[0]
        nbest_score=nbest_score.view(-1)
        nbest_score, best_idx=nbest_score.topk(beam,dim=0, largest=True, sorted=True)
        prev_k=best_idx//V
        last_pred=best_idx%V
        nbest=nbest[:,prev_k]
        nbest=torch.cat([nbest,last_pred.unsqueeze(0)],dim=0)
        lm_state=lm_state[prev_k]
        lm_logit=lm_logit[prev_k]
        last_ct=last_ct[prev_k]
        emb=emb[prev_k]
        
        nonblank=last_pred!=blank_id
        if nonblank.sum()>0:
            emb2=model.emb(last_pred[nonblank].long())
            emb[nonblank]=emb2
            lm_state[nonblank]=torch.cat([lm_state[nonblank,:,1:], emb2[:,:,None]], dim=-1)
            lm[nonblank]=model.lm(lm_state[nonblank]).squeeze(-1)
            lm_logit[nonblank]=model.lm_out(lm[nonblank])
            last_ct[nonblank]=enc[k][nonblank]
        
    return nbest.permute(1,0),nbest_score

def test():
    total_l=0
    total_d=0
    total_i=0
    total_r=0
    total_s=0
    cv_dataset = SpeechDataset("data/aishell/aishell_test.csv".split(','),'test')
    cv_dataloader = DataLoader(cv_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn,drop_last=False,pin_memory=False)
    for i, minibatch in enumerate(cv_dataloader):
        x,y = minibatch
        with torch.no_grad():
            nbest,score=infer(x.cuda())
        lab=i2s(y[0])
        rec=i2s(nbest[0])
        d,ins,dele,sub,l = wer(lab, rec)
        total_l+=l
        total_d+=d
        total_i+=ins
        total_r+=dele
        total_s+=sub
        print(i)
        print('LAB:',lab)
        print('REC:',rec)
        print('WER:',d/l)
        print('mean wer:%.2f, I:%.2f D:%.2f S:%.2f'%(total_d/total_l*100,total_i/total_l*100,total_r/total_l*100,total_s/total_l*100))
        print('')
        if nbest_path:
            for r, s in zip(nbest, score):
                r=i2s(r)
                f.write(r+','+str(s.item())+'\n')
            f.write(lab+'\n')

if __name__ == "__main__":
    start_time=timeit.default_timer()
    test()
    time=timeit.default_timer()-start_time
    print('Time:',int(time//60),'m',int(time%60),'s')
    if nbest_path:
        f.close()
