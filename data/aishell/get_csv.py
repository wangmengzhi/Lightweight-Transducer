import csv
import os
import torchaudio

data={}
with open('data/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt',encoding='utf-8') as f:
    for l in f:
        l=l.split()
        data[l[0]]=' '.join(l[1:])

no_label=[]
no_audio=list(data.keys())

def get_csv(csvpath, datapath, name):
    print(name)
    cnt=0
    w=csv.writer(open('%s.csv'%(csvpath+name),'w',encoding='utf-8', newline=''))
    w.writerow(["path", "duration", "transcript"])
    for (dirpath, dirnames, filenames) in os.walk(datapath+name):
        for filename in filenames:
            if filename.lower().endswith('.wav'):
                filepath = os.sep.join([dirpath, filename])
                filename = filename.split('.')[0]
                if filename not in data:
                    no_label.append(filename)
                else:
                    cnt+=1
                    if cnt%1000==0:print(cnt)
                    info = torchaudio.info(filepath)
                    duration = round(info.num_frames / info.sample_rate, 2)
                    w.writerow([filepath, duration, data[filename]])
                    no_audio.remove(filename)

for name in ['train','dev','test']:
    get_csv('data/aishell/aishell_', 'data/aishell/data_aishell/wav/', name)
print('no_label', no_label)
print('no_audio', no_audio)
