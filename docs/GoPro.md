# Instruction for Image Deblurring on the GoPro Dataset

## Data Preparation
#### 1. Download [the train set](https://drive.google.com/file/d/1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI/view) and [the evaluation data](https://drive.google.com/file/d/1abXSfeRGrzj2mQ2n2vIBHtObU6vXvr7C/view). (The data in these links is provided by NAFNet.)
   
After downloading, it should be like this:
```
./datasets/
└── GoPro/
    ├── train/
    |   ├── input/
    |   └── target/
    └── test/
        ├── input.lmdb
        └── target.imdb
```
#### 2. Preprocess the train set by this:

``` 
python scripts/data_preparation/gopro.py
```

This crops the train image pairs to 512x512 patches and makes the data into lmdb format.

## Training
 ``` 
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt options/train/GoPro/SWSformer.yml --launcher pytorch
```

## Evaluation
Set the path to the trained model weights in [the config file](../options/test/GoPro/SWSformer.yml)
 ``` 
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/GoPro/SWSformer.yml --launcher pytorch
```

