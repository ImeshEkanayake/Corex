# FairDP: Certified Fairness with Differential Privacy

## 1. Requirements:
- pandas 1.3.5  
- numpy 1.21.5
- scikit-learn 1.0.2
- tqdm 4.64.1
- matplotlib 3.5.3
- scipy 1.7.3
- pytorch 1.12.1
- torchvision 0.13.1
- torchaudio 0.12.1 
- cudatoolkit 11.3

Please run the following command to create the necessary folders 
```angular2html
bash shell/bash.sh
```

## 2. Data 

- For the `Adult` dataset, please download the data from: https://archive.ics.uci.edu/ml/datasets/adult
- For the `Default Credit Card Client` dataset, please download from: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
- For the `UTK-Face` dataset, please download from: https://susanqq.github.io/UTKFace/

## 2. Running process:

### 2.1 Important Parameters:
The parameters are parsed in the `main.py` through the `config.py` file. The uses of the paramerters
are as follows:
```angular2html
--dataset [DATASET ('adult', 'abalone', 'lawschool', 'ccc')]
--model_type [('NN', 'LR')]
--lr [learning rate)]
--sampling_rate [sampling rate q for FairDP/DPSGD/DPSGDF/DPSGD-Smooth)]
--n_layer [# hidden layer of NN)]
--n_hid [hidden dimension of NN)]
--clip [clipping bound C]
--optimizer [OPTIMIZER ]
--epochs [number of updating steps]
--ns [noise scale sigma for FairDP]
--performance_metric [PERFORMANCE_METRIC ]
--tar_eps [target privacy budget (FM/FairFM/FairFM-Smooth)]
--clip_layer [weight clipping hypyer parameter M]
```
### 2.2 Running
#### Clean
```
for RUN in 1 2 3 4 5
do
    python main.py --mode clean \
        --dataset <dataset> \
        --lr <learning_rate> \
        --batch_size <batch_size> \
        --model_type <model_type> \
        --n_layer <num_layer> \
        --n_hid <hidden_dimension> \
        --epochs <num_epochs> \
        --performance_metric <performance_metric> \
        --seed $RUN
done
```
#### FairDP
```
for RUN in 1 2 3 4 5 
do
    python main.py --mode lipr_fairdp_sum \
            --dataset <dataset> \
            --lr <learning_rate> \
            --batch_size <batch_size> \
            --sampling_rate <sampling rate in one batch> \
            --model_type <model_type> \
            --n_layer <number of layer> \
            --n_hid <hidden layer dimension> \
            --epochs <number of updating step> \
            --seed $RUN \
            --clip <gradient clipping parameter> \
            --ns <noise scale for DP preservation> \
            --performance_metric <metrics for performance> \
            --clip_layer <last layer's weights clipping parameter>
done

```
#### FairFM
```
for run in 1 2 3 4 5
do 
    python main.py --mode func \
        --submode func \
        --dataset <dataset> \
        --lr <learning rate> \
        --epochs <num epochs> \
        --performance_metric <metrics for performance> \
        --seed $run \
        --tar_eps <targeted prviacy budget>
done
```
#### FairFM-Smooth
```
for run in 1 2 3 
do 
    python main.py --mode func \
        --submode func_org \
        --dataset <dataset> \
        --lr <learning rate> \
        --epochs <num epochs> \
        --num_draws <number of time drawing noise for smooth classifier> \
        --performance_metric <metrics for performance> \
        --seed $run \
        --tar_eps <targeted prviacy budget>
done
```

