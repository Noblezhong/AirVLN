
# conda activate AirVLN

# cd ./AirVLN
echo $PWD

# 遵循TF模式粗略训练出一个“入门” policy network
python -u ./src/vlnce_src/train.py \
--run_type train \
--policy_type seq2seq \
--collect_type TF \
--name AirVLN-seq2seq \
--batchSize 4 \
--dagger_it 1 \
--epochs 500 \
--lr 0.00025 \
--trainer_gpu_device 0



nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0 &

# 以Dagger方法训练出policy network（从零开始）
python -u ./src/vlnce_src/dagger_train.py \
--run_type train \
--policy_type seq2seq \
--collect_type dagger \
--name AirVLN-seq2seq-dagger \
--batchSize 4 \
--dagger_it 10 \
--epochs 5 \
--lr 0.00025 \
--trainer_gpu_device 0 \
--dagger_update_size 5000


