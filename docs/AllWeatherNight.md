# Instruction for Multi-weather Nighttime image restoration on the AllWeatherNight Dataset

## Data Preparation
#### Download [the train set](https://drive.google.com/file/d/1EHW28JwZRrh_KuebBJuRwMwK9ilQpqmT/view) and [the evaluation data](https://drive.google.com/file/d/1-ay6OeSHTjVreNKGI3X52JuxTXoRK5kF/view). 
  
After downloading, it should be like this:
```
./datasets/
└── AllWeatherNight/
    ├── train/
    |   ├── rain_scene/
    |   ├── rain_scene_gt/
    |   ├── snow_scene/
    |   └── snow_scene_gt/
    └── test/
        ├── rain_scene/
        ├── rain_scene_gt/
        ├── snow_scene/
        └── snow_scene_gt/
```

## Training
* Rain scene
 ``` 
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt options/train/AllWeatherNight/rain_SWSformer.yml --launcher pytorch
```

* Snow scene
 ``` 
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt options/train/AllWeatherNight/snow_SWSformer.yml --launcher pytorch
```

## Evaluation
Set the path to the trained model weights in the config file.
* Rain scene
 ``` 
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/AllWeatherNight/rain_SWSformer.yml --launcher pytorch
```

* Snow scene
 ``` 
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/AllWeatherNight/snow_SWSformer.yml --launcher pytorch
```
