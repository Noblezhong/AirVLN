
# conda activate AirVLN

# cd /home/yons/zt/code/AirVLN_ws/AirVLN
echo $PWD


nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0 &

python -u ./src/vlnce_src/train.py \
--run_type collect \
--policy_type seq2seq \
--collect_type TF \
--name AirVLN-seq2seq \
--batchSize 2



