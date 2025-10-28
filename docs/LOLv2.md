# Instruction for Low-light enhancement on the LOLv2 Dataset

## Data Preparation
#### Download [the train set](https://drive.google.com/file/d/1Mx4eZgkoZNAvyI9QsyLxQGwI7vUzXFAm/view) and [the evaluation data](https://drive.google.com/file/d/1HJQS24ho6OPV5hGFOj3v3qUl6AK5kFiZ/view). 
  

After downloading, it should be like this:
```
./datasets/
└── LOLv2/
    ├── train/
    |   ├── Low/
    |   └── Normal/
    └── test/
        ├── Low/
        └── Normal/
```

## Training
 ``` 
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt options/train/LOLv2/SWSformer.yml --launcher pytorch
```

## Evaluation
Set the path to the trained model weights in [the config file](../options/test/LOLv2/SWSformer.yml).
 ``` 
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/LOLv2/SWSformer.yml --launcher pytorch
```

