device=0

# Data
data="adult"
dmode="rat"
data_path="Data/"
# rat=-1

# General
gmode="fairdp"
pname="${data}-${gmode}"
model_path="results/models/"
res_path="results/dict/"

# Model
model=NN
lr=0.007
bs=256
nhid=8
nlay=2
opt="adam"
epochs=5
dout=0.0
debug=0

#DP
srate=0.01
cgrad=1.0
n_mo=10
clay=0.76
ns=1.0

for rat in 1.0 1.25 1.5
do
    for epsilon in 0.5 1.5 2.0
    do
        pname="${data}-${gmode}-eps-${epsilon}-rat-${rat}"
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
                --epsilon $epsilon \
                --cgrad $cgrad \
                --srate $srate \
                --ns $ns \
                --n_mo $n_mo \
                --clay $clay \
                --debug $debug \
                --seed $run 
        done
    done
done



# Data
data="ccc"
clay=0.4

for rat in 1.0 1.25 1.5
do  
    for epsilon in 0.5 1.5 2.0
    do
        pname="${data}-${gmode}-eps-${epsilon}-rat-${rat}"
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
                --epsilon $epsilon \
                --cgrad $cgrad \
                --srate $srate \
                --ns $ns \
                --n_mo $n_mo \
                --clay $clay \
                --debug $debug \
                --seed $run 
        done
    done
done


data="utk"
model=CNN
clay=0.7


for rat in 1.0 1.25 1.5
do
    for epsilon in 0.5
    do
        pname="${data}-${gmode}-eps-${epsilon}-rat-${rat}"
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
                --epsilon $epsilon \
                --cgrad $cgrad \
                --srate $srate \
                --ns $ns \
                --n_mo $n_mo \
                --clay $clay \
                --debug $debug \
                --seed $run 
        done
    done
done

clay=0.5


for rat in 1.0 1.25 1.5
do
    for epsilon in 1.5 2.0
    do
        pname="${data}-${gmode}-eps-${epsilon}-rat-${rat}"
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
                --epsilon $epsilon \
                --cgrad $cgrad \
                --srate $srate \
                --ns $ns \
                --n_mo $n_mo \
                --clay $clay \
                --debug $debug \
                --seed $run 
        done
    done
done
