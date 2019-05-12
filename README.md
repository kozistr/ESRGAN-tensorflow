# ESRGAN in tensorflow
Enhanced Super Resolution Generative Adversarial Network in tensorflow

This repo is based on pytorch impl [original here](https://github.com/xinntao/ESRGAN)

**Work In Process :)**

# Requirements

* python 2.x / 3.x
* tensorflow-gpu 1.x
* opencv
* glob
* tqdm

# Repo-Tree

# Usage

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
