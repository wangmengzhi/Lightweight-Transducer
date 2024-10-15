import os
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import random
import soundfile
import timeit
import argparse
import pandas
from torch.cuda.amp import autocast as autocast
from conformer import ConformerBlock
from conv_stft import STFT
import scipy.signal

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark=True

n_class=4233
cur_step=0

def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.)

def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = np.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank

def audiofile_to_input_vector(audio_filename, train=True):
    try:
        x,fs=soundfile.read(audio_filename)
        x*=32768
    except Exception as e:
        print(e,audio_filename)
        return None,16000
    
    if train and (len(x)>20*fs or len(x)<0.2*fs):return None,16000
    if fs!=16000:
        if fs==8000:
            pass
        elif fs%16000==0:
            x=x[::fs//16000]
            fs=16000
        else:
            x = scipy.signal.resample(x, int(len(x)/fs*16000))
            fs=16000
    return x,fs

def get_dict():
    w2i={}
    i2w=[]
    f = open('vocab.txt', "r",encoding='utf-8')
    s = f.readlines()
    f.close()
    for l in s:
        l=l.strip()
        if l not in w2i:
            w2i[l]=len(w2i)
            i2w.append(l)
    return w2i,i2w

w2i,i2w=get_dict()

sos_id=w2i['<s>']
eos_id=w2i['</s>']
unk_id=w2i['<unk>']

def text_to_char_array(text):
    label=[]
    for c in text.lower():
        if c in w2i:
            label.append(w2i[c])
        elif c!=' ':
            label.append(unk_id)
    label.append(eos_id)
    return np.asarray(label)

def collate_fn(batch):
    xs=[]
    ys=[]
    xl=[]
    yl=[]
    srs=[]
    
    batch_size=len(batch)
    for i,(x,y,sr) in enumerate(batch):
        xl.append(len(x))
        yl.append(len(y))
        xs.append(x)
        ys.append(y)
        srs.append(sr)
    xs_pad=np.full((batch_size,np.max(xl)),0.0)
    ys_pad=np.full((batch_size,np.max(yl)),-1)
    x_mask=np.full((batch_size,np.max(xl)),0)
    for i in range(batch_size):
        xs_pad[i,:xl[i]]=xs[i]
        x_mask[i,:xl[i]]=1
        ys_pad[i,:yl[i]]=ys[i]
    
    return torch.FloatTensor(xs_pad).contiguous(), torch.IntTensor(ys_pad).contiguous(), torch.FloatTensor(x_mask).contiguous(), torch.IntTensor(srs).contiguous()

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
        self.files=self.files.sort_values(by="wav_filesize").loc[:, ["wav_filename", "transcript"]].values #[::-1]
        
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
        wav_file, transcript=self.files[i]
        x,sr = audiofile_to_input_vector(wav_file, self.name=='train')
        y = text_to_char_array(transcript)
        if x is None or y is None:
            x=np.zeros(int(16000*0.2))
            y=np.asarray([eos_id])
        return x,y,sr

class Encoder(nn.Module):
    def __init__(self,dmodel,ctc_out):
        super(Encoder, self).__init__()
        n_input=80
        self.dropout_rate=0.1
        self.dropout=nn.Dropout(self.dropout_rate,inplace=True)
        self.bn=torch.nn.BatchNorm1d(n_input,affine=False)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2,1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2,1),
            nn.LeakyReLU(inplace=True),
        )
        self.proj=nn.Linear(n_input//4*64,dmodel)
        net=[]
        self.n_layers=12
        for i in range(self.n_layers):
            block = ConformerBlock(
                dim = dmodel,
                dim_qk=64,
                dim_v = 64,
                heads = dmodel//64,
                ff_mult = 2048//dmodel,
                lorder =7,
                rorder =7,
                attn_dropout = self.dropout_rate,
                ff_dropout = self.dropout_rate,
                conv_dropout = self.dropout_rate,
                att_mask=None
            )
            net.append(block)
        self.net = nn.Sequential(*net)
        self.down=nn.Conv1d(dmodel,dmodel,2,2,0)

    def forward(self, x, x_mask):
        x=x.permute(0,2,1)
        x=self.bn(x)
        if self.training:
            for i in range(2):
                f=random.randint(0,10)
                f0=random.randint(0,x.shape[1]-f)
                x[:,f0:f0+f]=0
                t=random.randint(0,50)
                t=min(t,int(x.shape[2]*0.3))
                t0=random.randint(0,x.shape[2]-t)
                x[:,:,t0:t0+t]=0
        x=x.unsqueeze(1)
        x=self.conv(x)
        x=x.view(x.shape[0],-1,x.shape[-1])
        x=x.permute(0,2,1) #B H T->B T H
        x=self.proj(x)
        x=self.dropout(x)
        for i in range(len(self.net)):
            x = self.net[i](x)
            if i==3:
                x=self.down(x.permute(0,2,1)).permute(0,2,1)
        return x.permute(1,0,2)

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.bce_loss=nn.BCELoss(reduction='none')
        
    def forward(self, logit, label, mask, one_hot,smooth,num_class,bce=False):
        label=label.clone()
        label[(label<0) | (label>=num_class)]=0
        if one_hot:
            label=torch.nn.functional.one_hot(label.long(),num_class)
        label=label*smooth+(1-smooth)/num_class
        if bce:
            label[:,:,0]=0
            loss=self.bce_loss(logit,label)
        else:
            loss=-torch.log_softmax(logit,dim=2)*label
        loss=loss.sum(dim=2)
        loss=loss*mask
        loss=loss.sum()/mask.sum().clamp(min=1)
        return loss

def ctc_fa(ctc_logit,blank_id,label):
    pad_value=torch.finfo(ctc_logit.dtype).min
    label=label.permute(1,0).long()
    label[label==blank_id]=-1
    B,T=label.shape
    yl=(label>=0).sum(1)*2+1
    label=torch.cat([label.new_zeros(label.shape)+blank_id,label],dim=-1).reshape(B,2,T).permute(0,2,1).reshape(B,-1)
    bi=torch.arange(B).to(label)
    label=F.pad(label,[0,1],value=blank_id)
    
    T=ctc_logit.shape[1]
    lab_T=label.shape[1]
    bi2=bi.unsqueeze(-1).repeat(1,lab_T).view(-1)
    
    dp=ctc_logit.new_zeros(B,T,lab_T)+pad_value
    dp[:,0,0]=ctc_logit[:,0,blank_id]
    dp[:,0,1]=ctc_logit[bi,0,label[:,1]]
    bp=torch.arange(lab_T).view(1,1,-1).repeat(B,T,1).to(ctc_logit).long()
    
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

class Model(nn.Module):
    def __init__(self, dmodel):
        super(Model, self).__init__()
        self.emb_dim=512
        self.bce_loss=nn.BCELoss()
        self.emb=nn.Embedding(n_class,self.emb_dim)
        self.context_size=2
        self.lm=nn.Sequential(
            nn.Conv1d(self.emb_dim,self.emb_dim,self.context_size),
            nn.LeakyReLU(inplace=True)
        )
        self.to_blank=nn.Sequential(
            nn.Linear(in_features=dmodel*2+self.emb_dim,out_features=dmodel,bias=True),
            nn.Tanh(),
            nn.Linear(in_features=dmodel,out_features=1,bias=True),
            nn.Sigmoid()
        )
        self.out=nn.Sequential(
            nn.Linear(in_features=dmodel+self.emb_dim,out_features=dmodel,bias=True),
            nn.Linear(in_features=dmodel,out_features=n_class,bias=True),
        )
        self.ctc_out=nn.Linear(in_features=dmodel,out_features=n_class,bias=True)
        self.encoder=Encoder(dmodel,self.ctc_out)
        self.loss = CELoss()
        self.stft = STFT(win_len=int(16000*0.032),win_hop=int(16000*0.01),fft_len=int(16000//31.25),pad_center=False,win_type='hamm')
        self.stft_8k = STFT(win_len=int(8000*0.032),win_hop=int(8000*0.01),fft_len=int(8000//31.25),pad_center=False,win_type='hamm')
        fb=get_filterbanks(80,512,16000,0,8000).T.astype(np.float32)
        self.fb = torch.from_numpy(fb)
        self.fb_inv=torch.from_numpy(np.linalg.pinv(fb))
        self.ctc_loss=nn.CTCLoss(blank=0,reduction='none', zero_infinity=True) #mean  
    
    def get_feat(self, x, x_mask, speedup,sr):
        if x_mask is None:
            x_mask=x.new_ones(x.shape[0],x.shape[1])
        
        x=x/32768.0
        
        total_scale=32768
        post_scale=4096
        pre_scale=total_scale/post_scale
        x=x*pre_scale
        x=torch.cat([x[:,:1],x[:,1:]-0.97*x[:,:-1]],dim=1)
        
        idx1=sr==16000
        idx2=sr==8000
        if idx1.sum()>0:
            x_16k, phase_16k = self.stft.transform(x[idx1], return_type='magphase')
        if idx2.sum()>0:
            x_8k, phase_8k = self.stft_8k.transform(x[idx2], return_type='magphase')
            if idx1.sum()>0:
                x=x_16k.new_zeros(x.shape[0],x_16k.shape[1],max(x_16k.shape[2],x_8k.shape[2]))
                x[idx1,:,:x_16k.shape[2]]=x_16k
                x[idx2,:x_8k.shape[1],:x_8k.shape[2]]=x_8k
            else:
                x=F.pad(x_8k,(0,0,0,257-129,0,0),'constant',value=0)
        else:
            x=x_16k
        x=x.permute(0,2,1)
        x_len=(x_mask.sum(1)-sr*0.032)//(sr*0.01)+1
        
        if self.training:
            if speedup is not None and speedup!=1:
                t=x.new_zeros(x.shape[0],int(x.shape[1]//speedup),257)
                idx=torch.arange(x.shape[1]//speedup).long()
                t[:,idx]=x[:,(speedup*idx).long()]
                x=t
                x_len=x_len//speedup
        
        nfft=(sr*0.032).reshape(-1,1,1)
        x=(x**2)/nfft
        x=torch.matmul(x,self.fb.to(x))
        x=torch.log10(torch.clamp(x,min=1/(post_scale**2)))+math.log10(post_scale**2)
        
        x_len= x_len.unsqueeze(1).expand(x.shape[0],x.shape[1])
        seq_range = torch.arange(0,x.shape[1]).unsqueeze(0).expand(x.shape[0], x.shape[1])
        x_mask=(seq_range.to(x_len) < x_len).float()
        
        return x.detach(),x_mask.detach()
        
    def forward(self, x,y=None,x_mask=None,speedup=None,sr=torch.tensor([16000])):
        with autocast():
            x,x_mask=self.get_feat(x,x_mask,speedup,sr)
            
        enc=self.encoder(x,x_mask)
        T,B,H=enc.shape
        
        ctc_logit=self.ctc_out(enc).log_softmax(-1)
        if y is not None:
            ctc_label=y.clone()
            ctc_label_len=(y>=0).sum(1)-1
            ctc_label[(ctc_label<0)|(ctc_label==eos_id)]=0
            ctc_in_len=torch.full((B,),T,dtype=torch.long)
            ctc_loss=self.ctc_loss(ctc_logit,ctc_label,ctc_in_len,ctc_label_len)
            ctc_loss=ctc_loss/ctc_label_len
            y2=y.clone()
            y2[y2==eos_id]=-1
            y2=y2.permute(1,0)
            y2_mask=(y2>=0).float()
            lab_T=y2.shape[0]
            
            ctc_align,ctc_loss2=ctc_fa(ctc_logit.permute(1,0,2),0,y2)
            ctc_align=ctc_align[:,:,1::2].permute(0,2,1) #B LAB_T T
            ctc_align_norm=ctc_align/(ctc_align.sum(-1,keepdim=True).clamp(min=1))
            
            ctc_align_cum=torch.cumsum(ctc_align,dim=-1)
            ctc_align=(ctc_align_cum==1)&(ctc_align>0)
            nonblank=(ctc_align.sum(1)>0).float()
            transducer_lab=ctc_align.float().argmax(1)
            batch_idx=torch.arange(B,device=enc.device).long().unsqueeze(1).repeat(1,T)
            transducer_lab=y2.permute(1,0)[batch_idx.view(-1),transducer_lab.view(-1)]
            transducer_lab=transducer_lab.reshape(B,T)*nonblank
            transducer_lab=transducer_lab.permute(1,0)
            am=torch.bmm(ctc_align_norm,enc.transpose(1,0)).transpose(1,0)
            lm_input=y2.long()
            sos=torch.ones((1,B),dtype=torch.long,device=enc.device)*sos_id
            lm=torch.cat((sos,lm_input[:-1]),dim=0)
            lm=F.embedding(lm.clamp(min=0).long(),self.emb.weight)
            lm=lm.permute(1,2,0)
            lm=F.pad(lm,[self.context_size-1,0])
            lm=self.lm(lm)
            lm=lm.permute(2,0,1)
            batch_idx=torch.arange(B,device=enc.device).long()
            dec_idx=enc.new_zeros(B).long()
            logits=[]
            blanks=[]
            last_ct=enc.new_zeros(B,H)
            for i in range(T):
                blank=torch.cat([enc[i],lm[dec_idx,batch_idx],last_ct],dim=-1).detach()
                blanks.append(blank)
                last_ct[nonblank[:,i].bool()]=enc[i][nonblank[:,i].bool()]
                dec_idx=dec_idx+nonblank[:,i].long()
            blanks=torch.stack(blanks) #TB
            blanks=self.to_blank(blanks)
            amlm=torch.cat([am,lm],dim=-1)
            logit=self.out(amlm)
            nt_loss=self.loss(logit,y2,y2_mask*(ctc_loss2<2).float().unsqueeze(0),True,0.9,n_class)
            blank_label=1-nonblank.detach().float()
            blank_loss=self.bce_loss(blanks.squeeze(-1).transpose(1,0).float(), blank_label).mean()
            dec=(torch.argmax(logit,-1)*y2_mask.long()).permute(1,0)
        if not self.training:
            last_pred=torch.ones(B,dtype=torch.long,device=enc.device)*sos_id
            emb=F.embedding(last_pred.long(),self.emb.weight)
            lm_state=F.pad(emb.unsqueeze(-1), [self.context_size-1,0])
            lm=self.lm(lm_state).squeeze(-1)
            dec=enc.new_zeros(B,T).long()+eos_id
            last_ct=enc.new_zeros(B,H)
            for i in range(T):
                blank=torch.cat([enc[i],lm,last_ct],dim=-1)
                amlm=torch.cat([enc[i],lm],dim=-1)
                blank=self.to_blank(blank)
                logit=self.out(amlm)
                logit=logit[:,1:].softmax(-1)*(1-blank)
                logit=torch.cat([blank,logit],dim=-1)
                last_pred=logit.argmax(-1)
                dec[:,i]=last_pred
                nonblank=last_pred>0
                if nonblank.sum()>0:
                    this_emb=F.embedding(last_pred[nonblank].long(),self.emb.weight)
                    lm_state[nonblank]=torch.cat([lm_state[nonblank,:,1:], this_emb[:,:,None]], dim=-1)
                    lm[nonblank]=self.lm(lm_state[nonblank]).squeeze(-1)
                    last_ct[nonblank]=enc[i][nonblank]
                    emb[nonblank]=this_emb
            
        dec_ctc=torch.argmax(ctc_logit,-1).permute(1,0)
        if y is not None:
            return nt_loss,ctc_loss,ctc_loss2,dec,dec_ctc,blank_loss
        return x,ctc_logit,dec_ctc,dec

def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    swap=False
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n
        swap=True

    current = list(range(n+1))
    dp=np.zeros((m+1,n+1))
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
        dp[i]=current.copy()
    
    ins=0
    dele=0
    sub=0
    i=m
    j=n
    while i>0 and j>0:
        if dp[i][j]==min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1]):
            i-=1
            j-=1
        elif dp[i][j]==dp[i-1][j-1]+1:
            i-=1
            j-=1
            sub+=1
        elif dp[i][j]==dp[i-1][j]+1:
            i-=1
            if not swap:
                ins+=1
            else:
                dele+=1
        else:
            j-=1
            if not swap:
                dele+=1
            else:
                ins+=1
    if i>0:
        if not swap:
            ins+=dp[i][j]
        else:
            dele+=dp[i][j]
    elif j>0:
        if not swap:
            dele+=dp[i][j]
        else:
            ins+=dp[i][j]
        
    return current[n],ins,dele,sub

def wer(original, result):
    original = original.split()
    result = result.split()
    if len(original)==0:
        if len(result)==0:return (0,0,0,0), float(len(original))
        return (1,len(result),0,0), float(len(original))
    elif len(result)==0:
        return (1,0,len(original),0), float(len(original))
    return levenshtein(original, result) , float(len(original))

def id2s(ids,print_eos=False,bpe=False,pinyin=False):
    dic= i2p if pinyin else i2w
    s=[]
    for j in ids:
        if j==0:continue
        if j==eos_id:
            if print_eos:
                s.append('</s>')
            else:
                break
        if j < len(dic):
            t=dic[j]
            if bpe and t[0] not in "'▁" and (ord(t[0])<ord('a') or ord(t[0])>ord('z') or t[-1] in '01234'):
                t='▁'+t
            s.append(t)
    if bpe:
        s=''.join(s).split('▁')
    return ' '.join(s)

def id2s_ctc(ids,ctc_logit=None,lm_logits=None,bpe=False,pinyin=False):
    dic= i2p if pinyin else i2w
    s=[]
    for i,j in enumerate(ids):
        if j!=0 and (i==0 or j!=ids[i-1]) and j < len(dic):
            if ctc_logit is None or len(s)>lm_logits.shape[0]-1:
                t=dic[j]
            else:
                ctc_logit=torch.log_softmax(ctc_logit,dim=1)
                idx=torch.argmax(ctc_logit[i]+lm_logits[len(s)])
                t=dic[idx]
            if t=='</s>':break
            if bpe and t[0] not in "'▁" and (ord(t[0])<ord('a') or ord(t[0])>ord('z') or t[-1] in '01234'):
                t='▁'+t
            s.append(t)
    if bpe:
        s=''.join(s).split('▁')
    return ' '.join(s)

def train_once(model,tr_dataloader,epoch,optimizer,total_step,args):
    model.train()
    global cur_step
    total_loss=0
    for i, minibatch in enumerate(tr_dataloader):
        cur_step+=1
        if args.lr_schedule=='warmup':
            real_step=math.ceil(cur_step/args.accum_grad)
            adjust_lr(optimizer, args.init_lr *args.warmup_step**0.5*min(real_step**-0.5,real_step*args.warmup_step**-1.5))
        
        t1 = timeit.default_timer()
        x,y,x_mask,sr = minibatch
        if x.shape[1]>20*16000:
            print('seq_len is too large, B:%d,T:%d, skip batch'%(x.shape[0],x.shape[1]))
            continue
        speedup=random.choice([0.9,1.0,1.1])

        try:
            x=x.cuda()
            y=y.cuda()
            x_mask=x_mask.cuda()
            sr=sr.cuda()
            nt_loss,ctc_loss,ctc_loss2,dec,dec_ctc,blank_loss = model(x,y,x_mask,speedup,sr)
            nt_loss=nt_loss.mean()
            ctc_loss2=ctc_loss2.mean()
            blank_loss=blank_loss.mean()
            ctc_loss=ctc_loss.mean()
            loss=nt_loss*0.7+ctc_loss*0.3+blank_loss
            loss=loss/args.accum_grad
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        except RuntimeError as e:
            if 'out of memory' in str(e) or 'CUDNN_STATUS_EXECUTION_FAILED' in str(e):
                print('out of memory, B:%d,T:%d,dec_T:%d, skip batch'%(x.shape[0],x.shape[1],y.shape[1]))
                try:
                    x=x.cpu()
                    y=y.cpu()
                    x_mask=x_mask.cpu()
                    sr=sr.cpu()
                except RuntimeError as e:
                    print(e)
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
                lab=id2s(y[j],bpe=True)
                rec=id2s(dec[j],bpe=True)
                rec_ctc=id2s_ctc(dec_ctc[j],bpe=True)
                (d,ins,dele,sub),l = wer(lab, rec)
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
                
        t1 = t2
    return total_loss/(i+1)

def test_once(model,cv_dataloader,epoch,bpe=True):
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
            x,y,x_mask,sr = minibatch
            x=x.cuda()
            y=y.cuda()
            x_mask=x_mask.cuda()
            sr=sr.cuda()
            
            nt_loss,ctc_loss,ctc_loss2,dec,dec_ctc,blank_loss= model(x,y,x_mask,sr=sr)
            nt_loss=nt_loss.mean()
            ctc_loss=ctc_loss.mean()
            ctc_loss2=ctc_loss2.mean()
            blank_loss=blank_loss.mean()
            loss=nt_loss*0.7+ctc_loss*0.3+blank_loss

            total_loss+=loss.item()
            for j in range(len(dec)):
                lab=id2s(y[j],bpe=bpe)
                rec_ctc=id2s_ctc(dec_ctc[j],bpe=bpe)
                rec=id2s(dec[j],bpe=bpe)
                (d,ins,dele,sub),l = wer(lab, rec)
                
                if j==0:
                    print('LAB:',lab)
                    print('CTC:',rec_ctc)
                    print('NT:',rec)
                total_d+=d
                total_l+=l
                total_sub+=sub
                total_ins+=ins
                total_dele+=dele
                (d,ins,dele,sub),l = wer(lab, rec_ctc)
                total_d_ctc+=d
                total_l_ctc+=l
                total_sub_ctc+=sub
                total_ins_ctc+=ins
                total_dele_ctc+=dele
            
            print("TEST epoch: %d step: %d loss: %.3f mean loss: %.3f time: %.3f per: %.3f wer:%.2f S=%.2f I=%.2f D=%.2f ctc wer:%.2f S_ctc=%.2f I_ctc=%.2f D_ctc=%.2f" % (epoch,i,loss,total_loss/(i+1),timeit.default_timer()-t1,(i+1)/len(cv_dataloader),total_d/total_l*100,total_sub/total_l*100,total_ins/total_l*100,total_dele/total_l*100,total_d_ctc/total_l_ctc*100,total_sub_ctc/total_l_ctc*100,total_ins_ctc/total_l_ctc*100,total_dele_ctc/total_l_ctc*100))
        
class PartSampler():
    def __init__(self, start, end) -> None:
        self.l=range(start,end)

    def __iter__(self):
        return iter(self.l)

    def __len__(self):
        return len(self.l)
    
def train():
    parser = argparse.ArgumentParser(description="recognition argument")
    parser.add_argument("--epoch", type=int, default=360)
    parser.add_argument("--test_epoch", type=int, default=10)
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--accum_grad", type=int, default=4)
    parser.add_argument("--dropout",type=float,default=0.1)
    parser.add_argument("--init_lr",type=float,default=0.0015)
    parser.add_argument("--lr_schedule",type=str,default='warmup')
    parser.add_argument("--warmup_step", type=int, default=25000)
    parser.add_argument("--dmodel", type=int, default=256)
    parser.add_argument("--save_path",type=str,default='save/')
    parser.add_argument("--random_seed", type=int, default=0)
    args = parser.parse_args()
    
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    random.seed(args.random_seed)
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    model = Model(args.dmodel)
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr,betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    lr=args.init_lr
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
    
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn,drop_last=False,pin_memory=False)
    cv_dataloader = DataLoader(cv_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn,drop_last=False,pin_memory=False)
    
    total_step=len(tr_dataloader)*args.epoch
    global cur_step
    cur_step=len(tr_dataloader)*init_epoch
    
    if init_step>0:
        tr_dataloader2 = DataLoader(tr_dataset, sampler=PartSampler((init_step-cur_step)*args.batch_size,len(tr_dataset)),batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn,drop_last=False,pin_memory=False)
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
        torch.save(model.module.state_dict(), args.save_path+'model.'+str(epoch))
        torch.save(optimizer.state_dict(), args.save_path+'optim.'+str(epoch))
    test_once(model,cv_dataloader,args.epoch-1)
    train_time=timeit.default_timer()-start_time
    print('Training time:',int(train_time//3600),'h',int(train_time%3600//60),'m',int(train_time%60),'s')

if __name__ == "__main__":
     train()
