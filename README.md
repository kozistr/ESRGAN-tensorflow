# ESRGAN in tensorflow
Enhanced Super Resolution Generative Adversarial Network in tensorflow

This repo is based on pytorch impl [original here](https://github.com/xinntao/ESRGAN)

**Work In Process :)**

[![Total alerts](https://img.shields.io/lgtm/alerts/g/kozistr/ESRGAN-tensorflow.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/kozistr/ESRGAN-tensorflow/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/kozistr/ESRGAN-tensorflow.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/kozistr/ESRGAN-tensorflow/context:python)

# Requirements

* python 2.x / 3.x
* tensorflow-gpu 1.x
* opencv
* glob
* tqdm

# Repo-Tree

```
│
├── output  (generated images)
│     ├── ...
│     └── xxx.png
├── tb_logs (tensorboard records)
│     ├── [unique id]
│     │     ├── *.ckpt
│     │     ├── *.tsv
│     │     ├── *.meta
│     │     └── ...
│     └── [unique id]
├── requirements.txt  (requirements)
├── readme.md         (explaination)
├── losses.py         (useful losses)
├── metrics.py        (useful metrics)
├── model.py          (ESRGAN model)
├── main.py           (trainer / inferener)
├── config.py         (global configurations)
├── tfutils.py        (useful TF utils)
├── utils.py          (image processing utils)
└── dataloader.py     (DataSet loader)
```

# Usage

1. Clone this github repo.
```
git clone https://github.com/kozistr/ESRGAN-tensorflow
cd ESRGAN-tensorflow
```

2. install required packages (if needed)
```
# with pip
python -m pip install -r requirements.txt

# with conda
conda install --yes --file requirements.txt
```

3. run scripts!

For training,

```python3 train.py```

For evaluation,

```python3 evaluate.py```

For inference,

```python3 inference.py --src test-lr.png --dst test-hr.png```

# Results

# Citation

```
@InProceedings{wang2018esrgan,
    author = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Loy, Chen Change},
    title = {ESRGAN: Enhanced super-resolution generative adversarial networks},
    booktitle = {The European Conference on Computer Vision Workshops (ECCVW)},
    month = {September},
    year = {2018}
}
```

# Author
HyeongChan kim / [kozistr](http://kozistr.tech)
