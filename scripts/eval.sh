
# conda activate AirVLN

# cd ./AirVLN
echo $PWD


nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0 &


# collect-type只管如何去收集训练轨迹，评估阶段用不到这个参数，但是我改成dagger会跑不通
python -u ./src/vlnce_src/train.py \
--run_type eval \
--policy_type seq2seq \
--collect_type TF \
--name AirVLN-seq2seq-dagger \
--batchSize 2 \
--EVAL_CKPT_PATH_DIR ../DATA/output/AirVLN-seq2seq-dagger/train/checkpoint/20250731-110056-627500/ \
--EVAL_DATASET train \
--EVAL_NUM -1

# 脚本文件原始指令，评估没有使用dagger技术训练的agent
# python -u ./src/vlnce_src/train.py \
# --run_type eval \
# --policy_type seq2seq \
# --collect_type TF \
# --name AirVLN-seq2seq \
# --batchSize 2 \
# --EVAL_CKPT_PATH_DIR ../DATA/output/AirVLN-seq2seq/train/checkpoint \
# --EVAL_DATASET train \
# --EVAL_NUM -1


#  20250731-110123-667069/

# 我明明用的dagger方式训练，为什么不能在评估里切换，反而只能用Teacher Forcing？