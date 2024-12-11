# Lightweight-Transducer
Official implementation of the Interspeech 2024 paper ["Lightweight Transducer Based on Frame Level Criterion"](https://www.isca-archive.org/interspeech_2024/wan24_interspeech.pdf).

## Install
- Python3
- `pip install -r requirements.txt`
- (Optional) Refer to https://kheafield.com/code/kenlm/ for installing KenLM.

## Usage
### Data preparation
Download [AISHELL-1](https://www.openslr.org/resources/33/data_aishell.tgz) and extract it to the directory `data/aishell`.
```bash
python data/aishell/get_csv.py
python data/aishell/get_vocab.py
```

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
| dev | 3.79  |  0.10  |  0.07  | 3.96 |
| test | 4.10  |  0.16  |  0.05  | 4.31 |

[Pre-trained model](https://pan.baidu.com/s/14wgoeAlAYC0Y7X0PkO9bsQ?pwd=h8s9)

## Rescore
### Data preparation
Download [resource_aishell](https://www.openslr.org/resources/33/resource_aishell.tgz) and extract it to the directory `data/aishell`.
```bash
python data/aishell/get_text.py
```

### Training
```bash
../kenlm/build/bin/lmplz -o 3 --text data/aishell/aishell_train.txt --arpa data/aishell/aishell_train.arpa -S 10% --interpolate_unigrams 0
```

[Pre-trained language model](https://drive.google.com/file/d/1xwyQGTs_41Dww3KL5jx0s4_sfM5QIWO1/view?usp=sharing)

### Rescore
```bash
python rescore.py
```

### Results
| Testset |   Sub  |  Del  | Ins  |  CER |
| :---: |:----: |:----: |:----: | :----: |
| dev | 3.61  |  0.10  |  0.07  | 3.78 |
| test | 3.82  |  0.16  |  0.04  | 4.03 |