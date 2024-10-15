# Lightweight-Transducer
Official implementation of the Interspeech 2024 paper ["Lightweight Transducer Based on Frame Level Criterion"](https://www.isca-archive.org/interspeech_2024/wan24_interspeech.pdf).

## Install
- Python3
- `pip install -r requirements.txt`

## Usage
### Data preparation
Download [AISHELL-1](https://www.openslr.org/resources/33/data_aishell.tgz) and extract it to the directory `data/aishell`.

### Train
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

## Results
| Testset |   Sub  |  Del  | Ins  |  CER |
| :---: |:----: |:----: |:----: | :----: |
| dev | 4.03  |  0.10  |  0.08  | 4.21 |
| test | 4.30  |  0.18  |  0.06  | 4.54 |

[Pre-trained model](https://drive.google.com/file/d/1GRGUp4zx2QNlWbKIaRmVGkJV_MauR4Kh/view?usp=sharing)
