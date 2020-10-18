# Compressed Sensing Using Full Dimensional, Approximate Priors


## Steps to reproduce the results
NOTE: Please run **all** commands from the root directory of the repository, i.e from ```cs-full/```

---

1. Run ```sh setup.sh```
1. Run ```source env/bin/activate```
1. Run ```sh ./utils/run_sequentially.sh```
1. The Jupyter notebook in ```src/metric.ipynb``` contains a notebook to view quantitave results.
1. For qualitative results, run ```python src/view_estimated_celeba.py```
1. All plots will be in ```results/```

## Instructions for running individual experiments
The scripts in ```scripts/``` contain the python commands required to run our experiments.

Example code for ML. Change ```num-measurements``` appropriately. 

```
python -u ./src/compressed_sensing.py  --dataset celebA  --input-type=full-input  --num-input-images 5  --batch-size 5   --measurement-type gaussian  --noise-std=4  --num-measurements=1000   --model-types=map  --mloss-weight 1.0  --ploss-weight=0  --gradient-noise-weight=0  --zprior-weight=0   --optimizer-type adam  --learning-rate=1e-2  --momentum 0.9  --max-update-iter=2000  --num-random-restarts=2   --save-images  --save-stats  --print-stats  --checkpoint-iter 1  --image-matrix 0
```
Example code for MAP. Change ```num-measurements``` appropriately and ```ploss-weight``` to change the lambda parameter in our experiments.

```
python -u ./src/compressed_sensing.py  --dataset celebA  --input-type=full-input  --num-input-images 5  --batch-size 5   --measurement-type gaussian  --noise-std=4  --num-measurements=1000   --model-types=map  --mloss-weight 1.0  --ploss-weight=0.001  --gradient-noise-weight=0  --zprior-weight=0   --optimizer-type adam  --learning-rate=1e-2  --momentum 0.9  --max-update-iter=2000  --num-random-restarts=2   --save-images  --save-stats  --print-stats  --checkpoint-iter 1  --image-matrix 0 
```
Example code for our algorithm. Change ```num-measurements``` appropriately,  ```ploss-weight``` to change the lambda parameter in our experiments, and ```gradient-noise-weight``` to vary norm of Gaussian noise added in the gradient.


```
python -u ./src/compressed_sensing.py  --dataset celebA  --input-type=full-input  --num-input-images 5  --batch-size 5   --measurement-type gaussian  --noise-std=4  --num-measurements=1000   --model-types=noisy  --mloss-weight 1.0  --ploss-weight=0.001  --gradient-noise-weight=1.0  --zprior-weight=0   --optimizer-type adam  --learning-rate=1e-2  --momentum 0.9  --max-update-iter=2000  --num-random-restarts=2   --save-images  --save-stats  --print-stats  --checkpoint-iter 1  --image-matrix 0
```
