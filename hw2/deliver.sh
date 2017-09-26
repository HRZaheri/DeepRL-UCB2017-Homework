# Produce experiments results for CartPole
python train_pg.py CartPole-v0 -n 100 -b 1000 -e 5 -dna --exp_name sb_no_rtg_dna
python train_pg.py CartPole-v0 -n 100 -b 1000 -e 5 -rtg -dna --exp_name sb_rtg_dna
python train_pg.py CartPole-v0 -n 100 -b 1000 -e 5 -rtg --exp_name sb_rtg_na
python train_pg.py CartPole-v0 -n 100 -b 5000 -e 5 -dna --exp_name lb_no_rtg_dna
python train_pg.py CartPole-v0 -n 100 -b 5000 -e 5 -rtg -dna --exp_name lb_rtg_dna
python train_pg.py CartPole-v0 -n 100 -b 5000 -e 5 -rtg --exp_name lb_rtg_na

# Produce plot from results
python plot.py data/sb_rtg_na data/sb_rtg_dna data/sb_no_rtg_dna
python plot.py data/lb_rtg_na data/lb_rtg_dna data/lb_no_rtg_dna
python plot.py data/sb_rtg_dna data/lb_rtg_dna

# Produce experiments results for InvertedPendulum-v1
python train_pg.py InvertedPendulum-v1 -n 100 -b 1500 -e 3 -rtg --exp_name ip_rtg_na --learning_rate 3e-2 --n_layers 2 --size 16 --seed 13
python plot.py data/ip_

# NN Baseline
python train_pg.py InvertedPendulum-v1 -n 100 -b 1500 -e 3 -rtg -bl --exp_name ip_bl_rtg_na --learning_rate 3e-2 --n_layers 2 --size 16 --seed 13

# Cheetah
# basic
python train_pg.py HalfCheetah-v1 -ep 150 --discount 0.9 --exp_name hc2x64 -n 100 -b 5000 -e 1 --learning_rate 5e-2 -rtg --n_layers 2 --size 64 --seed 17
# tune
python train_pg.py HalfCheetah-v1 -ep 150 --discount 0.9 --exp_name hc2x32x15000x2e2 -n 100 -b 50000 -e 5 --learning_rate 4e-2 -rtg --n_layers 2 --size 32 --seed 17