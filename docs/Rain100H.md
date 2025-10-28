# Instruction for Image Deraining on the Rain100H Dataset

## Data Preparation
#### Download [the train set](https://drive.google.com/file/d/1azEjQDgSsz0AiofX_cIxAPre0bjL3wDi/view) and [the evaluation data](https://drive.google.com/file/d/1iz6sesYX9Zs12LFW10v_YVA1Q_FpIkIu/view). 
   
After downloading, it should be like this:
```
./datasets/
└── Rain100H/
    ├── train/
    |   ├── data/
    |   └── gt/
    └── test/
        ├── data/
        └── gt/
```

## Training
 ``` 
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt options/train/Rain100H/SWSformer.yml --launcher pytorch
```

## Evaluation
Set the path to the trained model weights in [the config file](../options/test/Rain100H/SWSformer.yml).
 ``` 
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/Rain100H/SWSformer.yml --launcher pytorch
```

