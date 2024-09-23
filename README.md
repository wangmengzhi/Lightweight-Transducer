# Lightweight-Transducer
The source code for the Interspeech 2024 paper ["Lightweight Transducer Based on Frame Level Criterion"](https://www.isca-archive.org/interspeech_2024/wan24_interspeech.pdf).

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
| dev | 4.12  |  0.12  |  0.07  | 4.30 |
| test | 4.47  |  0.18  |  0.07  | 4.73 |

[Pre-trained model](https://drive.google.com/file/d/1yvlIXrgV5GeWGWGsP_FkWMbQezm8x_Cg/view?usp=sharing)
