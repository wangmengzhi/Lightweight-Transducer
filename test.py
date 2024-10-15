import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from train import Model,audiofile_to_input_vector,id2s,id2s_ctc,get_dict,SpeechDataset,collate_fn,wer
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

w2i,i2w=get_dict()

model = Model(256)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
total_params = count_parameters(model)/1024/1024
print(f'Total parameters: {total_params}M')

model.load_state_dict(torch.load('save/model.avg'))
device = torch.device("cuda:0")
model.to(device)
model.eval()

beam=10
sos_id=w2i['<s>']
eos_id=w2i['</s>']
blank_id=0
logzero=-1e10
    
def infer(x, is_file,sr):
    if is_file:
        x,sr=audiofile_to_input_vector(x,False)
        x=torch.FloatTensor(x).unsqueeze(0).to(device)
        sr=torch.IntTensor([sr]).to(device)
    x,x_mask=model.get_feat(x,None,None,sr=sr)
    x=model.encoder(x,x_mask)
    T,_,H=x.shape
    
    last_pred=x.new_ones(beam,dtype=torch.long)*sos_id
    emb=F.embedding(last_pred.long(),model.emb.weight)
    lm_state=F.pad(emb.unsqueeze(-1), [model.context_size-1,0])
    lm=model.lm(lm_state).squeeze(-1)
    nbest_score=x.new_zeros(beam)
    nbest=x.new_zeros(0,beam).long()

    enc=x.repeat(1,beam,1)
    last_ct=x.new_zeros(beam,H)
    for k in range(T):
        blank=torch.cat([enc[k],lm,last_ct],dim=-1)
        amlm=torch.cat([enc[k],lm],dim=-1)
        blank=model.to_blank(blank)
        logit=model.out(amlm)
        logit=logit[:,1:].softmax(-1)*(1-blank)
        logit=torch.cat([blank,logit],dim=-1)
        logit=logit.clamp(min=1e-10).log()
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
        emb=emb[prev_k]
        last_ct=last_ct[prev_k]
        
        nonblank=last_pred!=blank_id
        if nonblank.sum()>0:
            this_emb=F.embedding(last_pred[nonblank].long(),model.emb.weight)
            lm_state[nonblank]=torch.cat([lm_state[nonblank,:,1:], this_emb[:,:,None]], dim=-1)
            lm[nonblank]=model.lm(lm_state[nonblank]).squeeze(-1)
            emb[nonblank]=this_emb
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
        x,y,x_mask,sr = minibatch
        x=x.cuda()
        with torch.no_grad():
            nbest,_=infer(x,False,sr.cuda())
        lab=id2s(y[0],bpe=True)
        rec=id2s(nbest[0],bpe=True)
        (d,ins,dele,sub),l = wer(lab, rec)
        total_l+=l
        total_d+=d
        total_i+=ins
        total_r+=dele
        total_s+=sub
        print(i)
        print('LAB:',lab)
        print('REC:',rec)
        print('WER:',d/l)
        print('mean wer:%.4f, I:%.4f D:%.4f S:%.4f'%(total_d/total_l,total_i/total_l,total_r/total_l,total_s/total_l))
        print('')

test()
