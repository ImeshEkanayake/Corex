device=0

# Data
data="adult"
dmode="none"
data_path="Data/"
rat=-1

# General
gmode="clean"
pname="${data}-${gmode}"
model_path="results/models/"
res_path="results/dict/"

# Model
model=NN
lr=0.005
bs=256
nhid=8
nlay=2
opt="adam"
epochs=50
dout=0.0
debug=0

for run in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=$device python main.py --pname $pname \
        --data $data \
        --gmode $gmode \
        --model_path $model_path \
        --res_path $res_path \
        --data_path $data_path \
        --dmode $dmode \
        --rat $rat \
        --model $model \
        --lr $lr \
        --bs $bs \
        --nhid $nhid \
        --nlay $nlay \
        --opt $opt \
        --epochs $epochs \
        --dout $dout \
        --debug $debug \
        --seed $run
done


# Data
data="ccc"
pname="${data}-${gmode}"

# Model
model=NN
lr=0.005
bs=256
nhid=8
nlay=2
opt="adam"
epochs=50

for run in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=$device python main.py --pname $pname \
        --data $data \
        --gmode $gmode \
        --model_path $model_path \
        --res_path $res_path \
        --data_path $data_path \
        --dmode $dmode \
        --rat $rat \
        --model $model \
        --lr $lr \
        --bs $bs \
        --nhid $nhid \
        --nlay $nlay \
        --opt $opt \
        --epochs $epochs \
        --dout $dout \
        --debug $debug \
        --seed $run
done


# # Data
# data="utk"
# pname="${data}-${gmode}"

# # Model
# model=CNN
# lr=0.001
# bs=256
# opt="adam"

# for run in 1 2 3 4 5
# do
#     CUDA_VISIBLE_DEVICES=$device python main.py --pname $pname \
#         --data $data \
#         --gmode $gmode \
#         --model_path $model_path \
#         --res_path $res_path \
#         --data_path $data_path \
#         --dmode $dmode \
#         --rat $rat \
#         --model $model \
#         --lr $lr \
#         --bs $bs \
#         --nhid $nhid \
#         --nlay $nlay \
#         --opt $opt \
#         --epochs $epochs \
#         --dout $dout \
#         --debug $debug \
#         --seed $run
# done