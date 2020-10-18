# Robust Compressed Sensing using Generative models

## Setup
---
1. Run ```git submodule init```
1. Run ```git submodule update```
1. Run ```python3.6 -m venv env```
1. Run ```source env/bin/activate```
1. Run ```pip install -r requirements.txt```

---

## Steps to reproduce the results
NOTE: Please run **all** commands from the root directory of the repository, i.e from ```csgm-robust-neurips/```


Example code for running MOM tournaments:
```
python src/compressed_sensing.py --net pggan-256 --dataset range --input-type full-input --num-input-images 1 --batch-size 1 --measurement-type dense --adversarial-epsilon 0.02 --noise-std 0.01 --num-measurements 1000 --model-types mom --mom-batch-size 20 --optimizer-type yellowfin --learning-rate 1e-5 --momentum 0. --max-update-iter 2000 --num-random-restarts 2 --save-images --save-stats --checkpoint-iter 1 --cuda --gif --gif-dir temp --gif-iter 10 
```
