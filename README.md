# Lightweight-Transducer
Official implementation of the Interspeech 2024 paper ["Lightweight Transducer Based on Frame Level Criterion"](https://www.isca-archive.org/interspeech_2024/wan24_interspeech.pdf).

## Install
- Python3
- `pip install -r requirements.txt`
- (Optional) Refer to https://kheafield.com/code/kenlm/ for installing KenLM.

## Usage
### Data preparation
Download [AISHELL-1](https://www.openslr.org/resources/33/data_aishell.tgz) and extract it to the directory `data/aishell`.

### Training
```bash
python train.py
```

### Parameter averaging
```bash
python avg_model.py
```

### Test
```bash
python test.py
```

### Speed
Training one epoch takes about five minutes with a single GPU RTX 4090 and CPU i9-13900K.

### Results
| Testset |   Sub  |  Del  | Ins  |  CER |
| :---: |:----: |:----: |:----: | :----: |
| dev | 3.93  |  0.11  |  0.07  | 4.10 |
| test | 4.14  |  0.19  |  0.06  | 4.39 |

[Pre-trained model](https://drive.google.com/file/d/1J3AEiurRSS0sjLqqcxiho0r-ykaqqotV/view?usp=sharing)

## Rescore
### Data preparation
```bash
python get_text.py
```

### Training
```bash
../kenlm/build/bin/lmplz -o 3 --text aishell_train.txt --arpa aishell_train.arpa -S 10% --interpolate_unigrams 0
```

[Pre-trained model](https://drive.google.com/file/d/1xwyQGTs_41Dww3KL5jx0s4_sfM5QIWO1/view?usp=sharing)

### Rescore
```bash
python rescore.py
```

### Results
| Testset |   Sub  |  Del  | Ins  |  CER |
| :---: |:----: |:----: |:----: | :----: |
| dev | 3.93  |  0.11  |  0.07  | 4.10 |
| test | 4.14  |  0.19  |  0.06  | 4.39 |