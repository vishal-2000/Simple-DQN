#!/bin/bash
#SBATCH --cpus-per-gpu=10
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-type=END
#SBATCH --output=op_file.txt
#SBATCH --time=4-00:00:00

source ~/anaconda3/etc/profile.d/conda.sh # Functions are not available by default in subshells, therefore this command is used to make conda functions available
conda activate vft
echo "Environment activated!!!"
cd /scratch
pwd
mkdir vishalmandadi_rl
cd ~
cp -R ~/Mid-Level-Planner /scratch/vishalmandadi_rl/
pwd
cd /scratch/vishalmandadi_rl/Mid-Level-Planner
pwd
echo "Training starts:"
python continue_trainer.py
# python train_wandb.py --experiment franka_panda --model delan --device cuda --model_save_path ./memory_shuffle/weights/model --train_file_path ./memory_shuffle/log_files/train_1.txt --weight-decay 0 --test_file_path ./memory_shuffle/log_files/test_1.txt --lr_decay_epochs 40000 --num-epochs 5000 --lr 0.0001 --number 1 --batch_size 32
echo "Finished training!"
echo "Creating a new folder in home for storing results"
cd ~
mkdir Results
echo "Copying results from /scratch to /home/Results"
cd /scratch
cp -R vishalmandadi_rl/Mid-Level-Planner/V2_next_best_action/models/model_checkpoints ~/Results/
echo "Results copied! A copy of results still exists in /scratch. Do not forget to remove it after checking the Results"
rm -r /scratch/vishalmandadi_rl
cd ~
