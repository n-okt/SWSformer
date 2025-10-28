# Instruction for Image Denoising on the SIDD Dataset

## Data Preparation
#### 1. Download [the train set](https://drive.google.com/file/d/1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw/view) and [the evaluation data](https://drive.google.com/file/d/1gZx_K2vmiHalRNOb1aj93KuUQ2guOlLp/view). (The data in these links is provided by NAFNet.)
   
After downloading, it should be like this:
```
./datasets/
└── SIDD/
    ├── Data/
    |   ├── 0001_.../
    |   ├── 0002_.../
    |   ...
    |
    └── val/
        ├── input_crops.imdb
        └── gt_crops.imdb
```
#### 2. Preprocess the train set by this:

``` 
python scripts/data_preparation/sidd.py
```

This crops the train image pairs to 512x512 patches and makes the data into lmdb format.

## Training
* 40 epochs
 ``` 
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt options/train/SIDD/SWSformer.yml --launcher pytorch
```
* 836 epochs

  (This experiment uses 3 GPUs. If using only 1 GPU, modify [the config file](../options/train/SIDD/SWSformer_largeEpochs.yml) and change the argument to ```--nproc_per_node=1```.)
 ``` 
python -m torch.distributed.launch --nproc_per_node=3 --master_port=4321 basicsr/train.py -opt options/train/SIDD/SWSformer_largeEpochs.yml --launcher pytorch
```

## Evaluation
Set the path to the trained model weights in [the config file](../options/test/SIDD/SWSformer.yml).
 ``` 
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/SIDD/SWSformer.yml --launcher pytorch
```


