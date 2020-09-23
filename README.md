# CarSharingTSP.jl
Surrogate model predicting TSP cost by means of neural network.

## Installation
Tested on Julia 1.5.1, but should work on older versions, too.
```
git clone https://github.com/dhonza/CarSharingTSP.jl.git

cd CarSharingTSP.jl.git

julia --project=.

julia> using Pkg; Pkg.instantiate()
```

## Run

Select training dataset in `src/carsharing-tsp.jl` (**TODO:** add command line options). Execute:

```
julia --project=. src/carsharing-tsp.jl
```

To run on RCI cluster:

```
cd slurm
sbatch train_gpu.batch
```
The trained models as well as train logs are stored in the `./exp` directory. RCI logs reside in `./logs`. 

## **TODO** add info on notebooks and conversion to Python
