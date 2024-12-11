import os
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import math
import timeit
import argparse
import pandas
from conformer import ConformerBlock

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark=True
    
def get_vocab():
    w2i={}
    i2w=[]
    with open('data/aishell/vocab.txt', "r",encoding='utf-8') as f:
        for l in f:
            l=l.strip()
            if l not in w2i:
                w2i[l]=len(w2i)
                i2w.append(l)
    return w2i,i2w

w2i,i2w=get_vocab()
sos_id=w2i['<s>']
eos_id=w2i['</s>']
unk_id=w2i['<unk>']
n_class=len(w2i)
cur_step=0

def get_audio(fp, train=True):
    try:
        x, sr=torchaudio.load(fp)
    except Exception as e:
        print(e,fp)
        return None
    if train and (x.shape[1]>20*sr or x.shape[1]<0.2*sr):return None
    x=x.mean(0)
    if sr!=16000:
        x=torchaudio.functional.resample(x, sr, 16000)
    return x

def s2i(text):
    label=[]
    for c in text.lower():
        if c in w2i:
            label.append(w2i[c])
        elif c!=' ':
            label.append(unk_id)
    label.append(eos_id)
    return torch.tensor(label, dtype=torch.int32)

class SpeechDataset(Dataset):
    def __init__(self, csvs, name,batch_size=None):
        self.name=name
        self.files = None
        for csv in csvs:
            f = pandas.read_csv(csv)
            if self.files is None:
                self.files = f
            else:
                self.files = self.files.append(f)
        self.files=self.files.sort_values(by="duration").loc[:, ["path", "transcript"]].values #[::-1]
        self.l=len(self.files)
        #if batch_size is not None:
            #self.shuffle(batch_size)
    
    def shuffle(self,batch_size):
        batch_num=self.l//batch_size
        l=batch_num*batch_size
        if l<self.l:
            remain=self.files[-(self.l-l):]
        nl=self.files
        self.files=[]
        mid=math.ceil(batch_num/2)
        for i in range(mid):
            self.files.extend(nl[i*batch_size:(i+1)*batch_size])
            j=batch_num-1-i
            if i!=j:
                self.files.extend(nl[j*batch_size:(j+1)*batch_size])
        if l<self.l:
            self.files.extend(remain)
        assert self.l==len(self.files)
    
    def __len__(self):
        return self.l
    
    def __getitem__(self, i):
        fp, transcript=self.files[i]
        x = get_audio(fp, self.name=='train')
        y = s2i(transcript)
        if x is None or y is None:
            x=torch.zeros(int(16000*0.2))
            y=torch.tensor([eos_id], dtype=torch.int32)
        return x,y

def collate_fn(batch):
    xs,ys= zip(*batch)
    xs=nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    ys=nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-1)
    return xs, ys

class Encoder(nn.Module):
    def __init__(self,dmodel):
        super(Encoder, self).__init__()
        n_input=80
        dropout_p=0.1
        self.dropout=nn.Dropout(dropout_p,inplace=True)
        self.bn=nn.BatchNorm1d(n_input,affine=False)
        self.aug = torchaudio.transforms.SpecAugment(n_time_masks=2, time_mask_param=50, n_freq_masks=2, freq_mask_param=10, iid_masks=True, p=0.3, zero_masking=True)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.proj = nn.Linear(n_input//4*64, dmodel)
        net=[]
        for i in range(12):
            block = ConformerBlock(
                dim = dmodel,
                dim_qkv = 64,
                dim_ff = 3072,
                kernel_size =15,
                dropout = dropout_p
            )
            net.append(block)
        self.net = nn.Sequential(*net)
        self.down = nn.Conv1d(dmodel, dmodel, 2, 2, 0)

    def forward(self, x):
        x=self.bn(x).unsqueeze(1)
        if self.training:
            x=self.aug(x)
        x=self.conv(x)
        x=x.view(x.shape[0],-1,x.shape[-1])
        x=x.transpose(1,2)
        x=self.proj(x)
        x=self.dropout(x)
        for i, block in enumerate(self.net):
            x = block(x)
            if i==3:
                x=self.down(x.transpose(1,2)).transpose(1,2)
        return x.transpose(0,1)

def ctc_fa(ctc_logit,blank_id,label):
    pad_value=torch.finfo(ctc_logit.dtype).min
    label=label.permute(1,0).long()
    label[label==blank_id]=-1
    B,U=label.shape
    yl=(label>=0).sum(1)*2+1
    U=U*2+1
    label2=torch.full((B, U), blank_id).to(label)
    label2[:,1::2]=label
    label=label2
    bi=torch.arange(B).to(label)
    
    T=ctc_logit.shape[1]
    bi2=bi.unsqueeze(-1).repeat(1,U).view(-1)
    
    dp=ctc_logit.new_zeros(B,T,U)+pad_value
    dp[:,0,0]=ctc_logit[:,0,blank_id]
    dp[:,0,1]=ctc_logit[bi,0,label[:,1]]
    bp=torch.arange(U).view(1,1,-1).repeat(B,T,1).to(ctc_logit).long()
    
    cond=(label==blank_id)|(label==F.pad(label,pad=[2,-2,0,0],value=-1))
    for i in range(1,T):
        a=torch.cat([dp[:,i-1].unsqueeze(-1),F.pad(dp[:,i-1],pad=[1,-1,0,0],value=pad_value).unsqueeze(-1)],dim=-1)
        v1,idx1=a.max(-1)
        a=torch.logsumexp(a,-1)
        a2=torch.cat([dp[:,i-1].unsqueeze(-1),F.pad(dp[:,i-1],pad=[1,-1,0,0],value=pad_value).unsqueeze(-1),F.pad(dp[:,i-1],pad=[2,-2,0,0],value=pad_value).unsqueeze(-1)],dim=-1)
        v2,idx2=a2.max(-1)
        a2=torch.logsumexp(a2,-1)
        dp[:,i]=ctc_logit[bi2,i,label.view(-1)].view(a.shape)+torch.where(cond,a,a2)
        bp[:,i]=bp[:,i]-torch.where(cond,idx1,idx2)
    ctc_loss=-torch.logaddexp(dp[bi,-1,yl-1],dp[bi,-1,yl-2])
    ctc_loss=2*ctc_loss/(yl-1)
    align=bp.new_zeros(bp.shape)
    wi=torch.where(dp[bi,-1,yl-1]>dp[bi,-1,yl-2],yl-1,yl-2)
    for i in range(T-1,-1,-1):
        align[bi,i,wi]=1
        wi=bp[bi,i,wi]
    return align,ctc_loss

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        
    def forward(self, logit, label, mask, smooth,num_class):
        label=label.clone()
        label[(label<0) | (label>=num_class)]=0
        label=torch.nn.functional.one_hot(label.long(),num_class)
        label=label*smooth+(1-smooth)/num_class
        loss=-torch.log_softmax(logit,dim=2)*label
        loss=(loss.sum(dim=2)*mask).sum()/mask.sum().clamp(min=1)
        return loss
    
class Model(nn.Module):
    def __init__(self, dmodel):
        super(Model, self).__init__()
        self.speed_perturb = torchaudio.transforms.SpeedPerturbation(16000, [0.9, 1.0, 1.1])
        self.fbank = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=80, window_fn=torch.hamming_window)
        self.encoder=Encoder(dmodel)
        self.ctc_out=nn.Linear(dmodel,n_class)
        self.am_out=nn.Linear(dmodel,n_class)
        self.context_size=2
        self.lm=nn.Sequential(
            nn.Conv1d(dmodel,dmodel,self.context_size),
            nn.LeakyReLU(inplace=True)
        )
        self.lm_out=nn.Linear(dmodel,n_class)
        self.blank=nn.Sequential(
            nn.Linear(dmodel*3,dmodel),
            nn.Tanh(),
            nn.Linear(dmodel,1)
        )
        self.loss = CELoss()
        self.bce_loss=nn.BCELoss()
        self.ctc_loss=nn.CTCLoss(blank=0,reduction='none', zero_infinity=True)
    
    def emb(self, x):
        return F.embedding(x, self.lm_out.weight)
        
    def get_feat(self, x):
        x=x*32768
        if self.training:
            x,_=self.speed_perturb(x)
        x=torchaudio.functional.preemphasis(x, coeff=0.97)
        x=self.fbank(x)
        x=x.clamp(min=1).log10()
        return x
        
    def forward(self, x,y=None):
        x=self.get_feat(x)
        enc=self.encoder(x)
        T,B,H=enc.shape
        ctc_logit=self.ctc_out(enc).log_softmax(-1)
        dec_ctc=torch.argmax(ctc_logit,-1).permute(1,0)
        if y is not None:
            ctc_label=y.clone()
            ctc_label[ctc_label==eos_id]=-1
            ctc_label_len=(ctc_label>=0).sum(1)
            ctc_in_len=torch.full((B,),T,dtype=torch.long)
            ctc_loss=self.ctc_loss(ctc_logit,ctc_label,ctc_in_len,ctc_label_len)
            ctc_loss=ctc_loss/ctc_label_len.clamp(min=1)
            ctc_label=ctc_label.permute(1,0)
            ctc_label_mask=(ctc_label>=0).float()
            ctc_align,ctc_loss2=ctc_fa(ctc_logit.permute(1,0,2),0,ctc_label)
            ctc_align=ctc_align[:,:,1::2].permute(0,2,1) #B U T
            ctc_align_norm=ctc_align/(ctc_align.sum(-1,keepdim=True).clamp(min=1))
            ctc_align_cum=torch.cumsum(ctc_align,dim=-1)
            ctc_align=(ctc_align_cum==1)&(ctc_align>0)
            nonblank=(ctc_align.sum(1)>0).float()
            am=torch.bmm(ctc_align_norm,enc.transpose(1,0)).transpose(1,0)
            emb=F.pad(ctc_label.long(),[0,0,1,-1],value=sos_id)
            emb=self.emb(emb.clamp(min=0).long())
            lm=emb.permute(1,2,0)
            lm=F.pad(lm,[self.context_size-1,0])
            lm=self.lm(lm)
            lm=lm.permute(2,0,1)
            lm_logit=self.lm_out(lm)
            logit=self.am_out(am)*(ctc_loss2<2).float()[None,:,None]+lm_logit
            nt_loss=self.loss(logit,ctc_label,ctc_label_mask,0.9,n_class)
            dec=(torch.argmax(logit,-1)*ctc_label_mask.long()).permute(1,0)
            batch_idx=torch.arange(B,device=enc.device).long()
            dec_idx=enc.new_zeros(B).long()
            blanks=[]
            last_ct=enc.new_zeros(B,H)
            for i in range(T):
                blank=torch.cat([enc[i],emb[dec_idx,batch_idx],last_ct],dim=-1)
                blanks.append(blank)
                last_ct[nonblank[:,i].bool()]=enc[i][nonblank[:,i].bool()]
                dec_idx=dec_idx+nonblank[:,i].long()
            blanks=torch.stack(blanks)
            blanks=self.blank(blanks.detach())
            blank_label=1-nonblank.detach().float()
            blank_loss=self.bce_loss(blanks.squeeze(-1).transpose(1,0).sigmoid(), blank_label)
        if not self.training:
            last_pred=torch.ones(B,dtype=torch.long,device=enc.device)*sos_id
            emb=self.emb(last_pred.long())
            lm_state=F.pad(emb.unsqueeze(-1), [self.context_size-1,0])
            lm=self.lm(lm_state).squeeze(-1)
            lm_logit=self.lm_out(lm)
            am_logit=self.am_out(enc)
            dec=enc.new_zeros(B,T).long()+eos_id
            last_ct=enc.new_zeros(B,H)
            for i in range(T):
                blank=torch.cat([enc[i],emb,last_ct],dim=-1)
                blank=self.blank(blank)
                logit=am_logit[i]+lm_logit
                blank2=F.logsigmoid(blank)
                logit=logit[:,1:].log_softmax(-1)+blank2-blank
                logit=torch.cat([blank2,logit],dim=-1)
                last_pred=logit.argmax(-1)
                dec[:,i]=last_pred
                nonblank=last_pred>0
                if nonblank.sum()>0:
                    emb2=self.emb(last_pred[nonblank].long())
                    emb[nonblank]=emb2
                    lm_state[nonblank]=torch.cat([lm_state[nonblank,:,1:], emb2[:,:,None]], dim=-1)
                    lm[nonblank]=self.lm(lm_state[nonblank]).squeeze(-1)
                    lm_logit[nonblank]=self.lm_out(lm[nonblank])
                    last_ct[nonblank]=enc[i][nonblank]
            
        if y is not None:
            return nt_loss,ctc_loss,ctc_loss2,dec,dec_ctc,blank_loss
        return x,am_logit,ctc_logit,dec,dec_ctc

def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def i2s(ids):
    s=[]
    for j in ids:
        if j==eos_id:
            break
        if j!=0:
            s.append(i2w[j])
    return ' '.join(s)

def i2s_ctc(ids):
    s=[]
    for i,j in enumerate(ids):
        if j!=0 and (i==0 or j!=ids[i-1]):
            s.append(i2w[j])
    return ' '.join(s)

def wer(reference, hypothesis):
    a = reference.split()
    b = hypothesis.split()
    m, n = len(a), len(b)

    dp=np.zeros((m+1,n+1))
    dp[range(m+1),0]=range(m+1)
    dp[0,range(n+1)]=range(n+1)

    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])+1
    
    I,D,S,i,j=0,0,0,m,n
    while i>0 and j>0:
        if dp[i][j]==min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1]):
            i-=1
            j-=1
        elif dp[i][j]==dp[i-1][j-1]+1:
            i-=1
            j-=1
            S+=1
        elif dp[i][j]==dp[i-1][j]+1:
            i-=1
            D+=1
        else:
            j-=1
            I+=1
    if i>0:
        D+=i
    elif j>0:
        I+=j
    return dp[m,n], I, D, S, float(m)

def train_once(model,tr_dataloader,epoch,optimizer,total_step,args):
    model.train()
    global cur_step
    total_loss=0
    for i, minibatch in enumerate(tr_dataloader):
        t1 = timeit.default_timer()
        cur_step+=1
        if args.lr_schedule=='noam':
            real_step=math.ceil(cur_step/args.accum_grad)
            adjust_lr(optimizer, args.lr *args.warmup_step**0.5*min(real_step**-0.5,real_step*args.warmup_step**-1.5))
        
        x,y = minibatch
        try:
            nt_loss,ctc_loss,ctc_loss2,dec,dec_ctc,blank_loss = model(x.cuda(),y.cuda())
            nt_loss=nt_loss.mean()
            ctc_loss=ctc_loss.mean()
            ctc_loss2=ctc_loss2.mean()
            blank_loss=blank_loss.mean()
            loss=nt_loss*0.7+ctc_loss*0.3+blank_loss
            loss=loss/args.accum_grad
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        except RuntimeError as e:
            if 'out of memory' in str(e) or 'CUDNN_STATUS_EXECUTION_FAILED' in str(e):
                print('out of memory, B:%d,T:%d,U:%d, skip batch'%(x.shape[0],x.shape[1],y.shape[1]))
                model.zero_grad()
                optimizer.zero_grad()
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad
                loss=None
                torch.cuda.empty_cache()
            else:
                raise e
            continue
        
        if cur_step%args.accum_grad==0 or i==len(tr_dataloader)-1:
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()
            
        t2 = timeit.default_timer()
        loss=loss*args.accum_grad
        total_loss+=loss.item()
        
        if cur_step%100==0:
            total_d=0.0
            total_l=0.0
            total_sub=0.0
            total_ins=0.0
            total_dele=0.0
            for j in range(len(dec)):
                lab=i2s(y[j])
                rec=i2s(dec[j])
                rec_ctc=i2s_ctc(dec_ctc[j])
                d,ins,dele,sub,l = wer(lab, rec)
                if j==0:
                    print('LAB:',lab)
                    print('CTC:',rec_ctc)
                    print('NT:',rec)
                total_d+=d
                total_l+=l
                total_sub+=sub
                total_ins+=ins
                total_dele+=dele
            total_l=max(total_l,1)
            
            print("TRAIN epoch: %d step: %d lr: %g  nt: %.3f ctc: %.3f ctc2: %.3f blank: %.3f loss: %.3f ml: %.3f time: %.3f per: %.3f wer:%.2f S=%.2f I=%.2f D=%.2f" % (epoch,cur_step,optimizer.param_groups[0]['lr'],nt_loss,ctc_loss,ctc_loss2,blank_loss,loss,total_loss/(i+1),t2-t1,cur_step/total_step,total_d/total_l*100,total_sub/total_l*100,total_ins/total_l*100,total_dele/total_l*100))
        else:
            print("TRAIN epoch: %d step: %d lr: %g  nt: %.3f ctc: %.3f ctc2: %.3f blank: %.3f loss: %.3f ml: %.3f time: %.3f per: %.3f" % (epoch,cur_step,optimizer.param_groups[0]['lr'],nt_loss,ctc_loss,ctc_loss2,blank_loss,loss,total_loss/(i+1),t2-t1,cur_step/total_step))
    return total_loss/(i+1)

def test_once(model,cv_dataloader,epoch):
    with torch.no_grad():
        model.eval()
        total_loss=0
        total_d=0.0
        total_l=0.0
        total_sub=0.0
        total_ins=0.0
        total_dele=0.0
        total_d_ctc=0.0
        total_l_ctc=0.0
        total_sub_ctc=0.0
        total_ins_ctc=0.0
        total_dele_ctc=0.0
        for i, minibatch in enumerate(cv_dataloader):
            t1 = timeit.default_timer()
            x,y = minibatch            
            nt_loss,ctc_loss,ctc_loss2,dec,dec_ctc,blank_loss= model(x.cuda(),y.cuda())
            nt_loss=nt_loss.mean()
            ctc_loss=ctc_loss.mean()
            ctc_loss2=ctc_loss2.mean()
            blank_loss=blank_loss.mean()
            loss=nt_loss*0.7+ctc_loss*0.3+blank_loss
            total_loss+=loss.item()

            for j in range(len(dec)):
                lab=i2s(y[j])
                rec_ctc=i2s_ctc(dec_ctc[j])
                rec=i2s(dec[j])
                d,ins,dele,sub,l = wer(lab, rec)
                if j==0:
                    print('LAB:',lab)
                    print('CTC:',rec_ctc)
                    print('NT:',rec)
                total_d+=d
                total_l+=l
                total_sub+=sub
                total_ins+=ins
                total_dele+=dele
                d,ins,dele,sub,l = wer(lab, rec_ctc)
                total_d_ctc+=d
                total_l_ctc+=l
                total_sub_ctc+=sub
                total_ins_ctc+=ins
                total_dele_ctc+=dele
            total_l=max(total_l,1)
            total_l_ctc=max(total_l_ctc,1)
            
            print("TEST epoch: %d step: %d loss: %.3f mean loss: %.3f time: %.3f per: %.3f wer:%.2f S=%.2f I=%.2f D=%.2f ctc wer:%.2f S_ctc=%.2f I_ctc=%.2f D_ctc=%.2f" % (epoch,i,loss,total_loss/(i+1),timeit.default_timer()-t1,(i+1)/len(cv_dataloader),total_d/total_l*100,total_sub/total_l*100,total_ins/total_l*100,total_dele/total_l*100,total_d_ctc/total_l_ctc*100,total_sub_ctc/total_l_ctc*100,total_ins_ctc/total_l_ctc*100,total_dele_ctc/total_l_ctc*100))
        return total_d/total_l*100
        
class PartSampler():
    def __init__(self, start, end) -> None:
        self.l=range(start,end)

    def __iter__(self):
        return iter(self.l)

    def __len__(self):
        return len(self.l)

def train():
    parser = argparse.ArgumentParser(description="recognition argument")
    parser.add_argument("--epoch", type=int, default=720)
    parser.add_argument("--test_epoch", type=int, default=10)
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--accum_grad", type=int, default=4)
    parser.add_argument("--lr",type=float,default=0.0015)
    parser.add_argument("--lr_schedule",type=str,default='noam')
    parser.add_argument("--warmup_step", type=int, default=25000)
    parser.add_argument("--dmodel", type=int, default=256)
    parser.add_argument("--save_path",type=str,default='save/')
    parser.add_argument("--random_seed", type=int, default=0)
    args = parser.parse_args()
    
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    model = Model(args.dmodel)
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())/1024/1024:.2f}M')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    tr_dataset = SpeechDataset("data/aishell/aishell_train.csv".split(','),'train',batch_size=args.batch_size)
    cv_dataset = SpeechDataset("data/aishell/aishell_test.csv".split(','),'test')
        
    init_epoch=0
    init_step=0
    last_model=None
    last_time=0
    for root,dirs,files in os.walk(args.save_path):
        for f in files:
            if f[:5]=='model' and f!='model.init' and f!='model.avg':
                t=os.path.getmtime(os.path.join(args.save_path,f))
                if t>last_time:
                    last_time=t
                    last_model=f
    if last_model is not None:
        t=last_model.split('.')[1]
        if '_' in t:
            init_epoch,init_step=t.split('_')
            init_epoch=int(init_epoch)
            init_step=int(init_step)
        else:
            init_epoch=int(t)
        model_path=args.save_path+last_model
        print('reading '+model_path)
        model.load_state_dict(torch.load(model_path))
        optimizer.load_state_dict(torch.load(args.save_path+last_model.replace('model.','optim.')))
            
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        print('restore '+model_path)
        if init_step==0:
            init_epoch+=1
        
    device = torch.device("cuda:0")
    model = nn.DataParallel(model)
    model.to(device)
    
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn,drop_last=False,pin_memory=False)
    cv_dataloader = DataLoader(cv_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn,drop_last=False,pin_memory=False)
    
    total_step=len(tr_dataloader)*args.epoch
    global cur_step
    cur_step=len(tr_dataloader)*init_epoch
    
    if init_step>0:
        tr_dataloader2 = DataLoader(tr_dataset, sampler=PartSampler((init_step-cur_step)*args.batch_size,len(tr_dataset)),batch_size=args.batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn,drop_last=False,pin_memory=False)
        cur_step=init_step
    print('step per epoch:',len(tr_dataloader))
    print('total step:',total_step)
    model.train()
    torch.save(model.module.state_dict(), args.save_path+"model.init")
    torch.save(optimizer.state_dict(), args.save_path+'optim.init')
    start_time=timeit.default_timer()
    for epoch in range(init_epoch,args.epoch):
        if epoch>0 and epoch%args.test_epoch==0 and epoch!=init_epoch:
            test_once(model,cv_dataloader,epoch-1)
        loss=train_once(model,tr_dataloader if init_step==0 or epoch!=init_epoch else tr_dataloader2,epoch,optimizer,total_step,args)
        print(f"Peak memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB, reserved: {torch.cuda.max_memory_reserved() / 1024 / 1024:.2f} MB")
        torch.save(model.module.state_dict(), args.save_path+'model.'+str(epoch))
        torch.save(optimizer.state_dict(), args.save_path+'optim.'+str(epoch))
    test_once(model,cv_dataloader,args.epoch-1)
    train_time=timeit.default_timer()-start_time
    print('Training time:',int(train_time//3600),'h',int(train_time%3600//60),'m',int(train_time%60),'s')

if __name__ == "__main__":
     train()
