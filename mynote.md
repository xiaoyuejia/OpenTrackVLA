# 运行指令
## 在备份目录 /data/hdt/newtrackvla修改/newtrackvla_base/ 下运行完整流水线：
```bash
cd /data/hdt/newtrackvla修改/newtrackvla_base

GPU0=0 \
GPU1=1 \
EVAL_NUM_GPUS=2 \
SIM_ROOT=/data/hdt/ntv_data/sim_data/data7_8 \
REUSED_DATA_ROOT=/data/hdt/ntv_data/data/data7_8_basevel_roi_train \
DATA_ROOT=/data/hdt/ntv_data/data/data7_8_recorded_pose_inversecontrol_roi_train_base \
CACHE_ROOT=/data/hdt/ntv_data/data/data7_8_recorded_pose_inversecontrol_roi_train_base/vision_cache \
OUT_DIR=/data/hdt/ntv_data/ckpt/data7_8_recorded_pose_inversecontrol_roi_bboxprompt_b64_2gpu_lr5e-5_base \
EVAL_ROOT=/data/hdt/ntv_data/sim_data/eval/data7_8_recorded_pose_inversecontrol_base_bbox0_100 \
TEST_MANIFEST=/data/hdt/ntv_data/sim_data/data7_8_bbox0_eval_100/split_manifest_100.json \
RUN_MAKE_DATA=1 \
RUN_TRAIN=1 \
RUN_EVAL=1 \
bash sh/run_data7_8_recorded_pose_inversecontrol_pipeline.sh
```
该流水线会依次执行：
从 data7_8 生成记录 pose、固定 0.1s waypoint 和逆控制标签；
复用已有 data7_8_basevel_roi_train 的 frames/vision cache；
在 GPU 0、1 上训练，每卡 batch 默认 32；
使用最新 checkpoint 在 GPU 0、1 并行评估；
评估保存视频、JSON、CSV 和汇总指标。

## 如果数据已经生成过，只重新训练和评估：
```bash
cd /data/hdt/newtrackvla修改/newtrackvla_base

GPU0=0 \
GPU1=1 \
EVAL_NUM_GPUS=2 \
DATA_ROOT=/data/hdt/ntv_data/data/data7_8_recorded_pose_inversecontrol_roi_train \
CACHE_ROOT=/data/hdt/ntv_data/data/data7_8_recorded_pose_inversecontrol_roi_train/vision_cache \
OUT_DIR=/data/hdt/ntv_data/ckpt/data7_8_recorded_pose_inversecontrol_roi_bboxprompt_b64_2gpu_lr5e-5_base \
EVAL_ROOT=/data/hdt/ntv_data/sim_data/eval/data7_8_recorded_pose_inversecontrol_base_bbox0_100 \
RUN_MAKE_DATA=0 \
RUN_TRAIN=1 \
RUN_EVAL=1 \
bash sh/run_data7_8_recorded_pose_inversecontrol_pipeline.sh
```
已有的 control_target_stats.json 位于：
/data/hdt/ntv_data/data/data7_8_recorded_pose_inversecontrol_roi_train/control_target_stats.json

我想现在只使用抽帧出来的图像，暂时不使用裁减出来的人的图像，在llm出来后的特征，再显式构建一个机器狗和无人机的cross-modal特征,先考虑使用transformer，后续再考虑使用其他模型。经过transformer，进行了decoder,再通过两个MLP，一个是bbox的头，和数据中的bbox做损失，一个是visibility的头，经过一个激活函数，做一个类似二分类的损失。我现在的输出waypoint的MLP头应该和transformer并列还是加在transformer后，以上是我的初步构思，可能不太合理，参考各种顶会、github现有的论文、仓库，讨论合理的方案


/data/hdt/ntv_data/data/data7_8_camera_m40_pose_fixed_dt_exact_bbox_train/ 我准备使用最新数据去训练


6 卡存在 3 个易失性 DRAM 不可纠正 ECC 错误、23 个不可纠正 retired pages，且 remapping
  已失败；这不是脚本问题，当前进程无法把 CUDA context 建在该卡上。为保证训练稳定，我先快速扫描其余空
  闲卡的可用性，给出可直接替代 6 卡的健康设备


cd /data/hdt/newtrackvla修改/newtrackvla_base
setsid -f env RENDER_GPU=6 WORKERS=1 \
RUNTIME_ROOT=/data/hdt/ntv_data/sim_data/unreal_env_workers/gpu7_replay \
OUTPUT_ROOT=/data/hdt/ntv_data/sim_data/keyboard_collect_inverse_replay_m40_m8_speed100_gpu7 \
MAX_RETRIES=20 SNAPSHOT_MODE=sequential \
VISUAL_LABEL_MODE=source_recording POST_RESET_SETTLE_S=8 \
WORKER_TIMEOUT_S=2400 \
bash sh/replay_keyboard_collect_gpu7_stable.sh \
>>/data/hdt/ntv_data/sim_data/keyboard_collect_inverse_replay_m40_m8_speed100_gpu7/replay_parent_gpu7.log 2>&1



启动多人数据处理
 cd /data/hdt/newtrackvla修改/newtrackvla_base && bash sh/preprocess_data7_29_dt_global_base_2gpu.sh
