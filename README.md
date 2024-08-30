# Language-driven Action Recognition

### Overview

Language-driven Action Recognition

### Installation

Option 1:

```
pip install -r requirements.txt
```

Option 2:

```
pip install torch
```

### Data Preparation

By default, for training, testing and demo, we use  [HAKE](https://github.com/DirtyHarryLYL/HAKE) dataset.

Instance-level part state annotations on [HICO-DET](http://www-personal.umich.edu/~ywchao/hico/) are also available.

The labels are packaged in **Annotations/hico-det-instance-level.tar.gz**, you could use:

```
cd Annotations
tar zxvf hico-det-instance-level.tar.gz
```

to unzip them and get hico-det-training-set-instance-level.json for train set of HICO-DET respectively. More details about the format are shown in [Dataset format](https://github.com/DirtyHarryLYL/HAKE/blob/master/Annotations/README.md).

The HICO-DET dataset can be found here: [HICO-DET](http://www-personal.umich.edu/~ywchao/hico/).

### Try demo now

By default, for test, we use demo.py 

```
python demo.py
```

By default, for training, we use train.py

```
python train.py
```

#### Model Zoo

| dataset    | fold | model | vit backbone | text encoder  | performance | url                                                 |
| ---------- | ---- | ----- | ------------ | ------------- | ----------- | --------------------------------------------------- |
| HAKE-train | 0    | LgAR  | ViT-B/32     | CLIP ViT-B/32 |             | [download](https://drive.google.com/file/d/1y4z4_yN |

If you find this repo useful, please cite [link]([Yangless/Language-driven-Action-Recognition: Language-driven Action Recognition (github.com)](https://github.com/Yangless/Language-driven-Action-Recognition)).

## Acknowledgement

Thanks to the code base from  [Pytorch](https://github.com/pytorch/pytorch),[CLIP](https://github.com/openai/CLIP),[HAKE](https://github.com/DirtyHarryLYL/HAKE).

