## 运行步骤
### 一、环境配置
####  1、使用配置好的conda环境
下载环境：https://pan.baidu.com/s/1YWSuoMlPPF8O6JXnHiNSLQ?pwd=mdvs -> 解压环境 -> 激活conda: activate openmmlab

打包conda环境（该指令打包时候用，配置时不需要！！！） ``` conda pack -n openmmlab -o openmmlab.tar.gz --ignore-editable-packages ```

#### 2、按照官网配置

### 二、基础配置
#### 1、基础配置更改
#####  配置1：configs/birenet/birenet_fcn_1xb4-306k_cityscapes-256x512.py
调整输入图片尺寸
crop_size = (512, 512)

原配置是4卡，改成单卡eta_min需要设置为原来的1/4
eta_min=1e-4 / 4,

原配置是4卡，改成单卡lr需要设置为原来的1/4
optimizer = dict(type='SGD', lr=0.05 / 4, momentum=0.9, weight_decay=0.0005)

添加mDice和mFscore指标

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'], output_dir='work_dir/birenet/format_results')
test_evaluator = val_evaluator

##### 配置2：configs/_base_/models/birenet.py
norm_cfg = dict(type='BN', requires_grad=True) #TODO SyncBN->BN
num_classes=2, #TODO 19->2

#####  配置3：configs/_base_/datasets/cityscapes_256x512.py
调整输入图片尺寸
crop_size = (256, 512)

#####  配置4：configs/_base_/datasets/cityscapes.py
调整数据集路径
data_root = ''

##### 配置5：configs/_base_/default_runtime.py
添加tensorboard和wandb
vis_backends=[dict(type='LocalVisBackend'),
              # tensorboard 路径eg:work_dir/birenet/20230510_200431/vis_data
              dict(type='TensorboardVisBackend'),
              # wandb 特征图路径eg：work_dir/birenet/20230511_150800/vis_data/wandb/run-20230511_150804-9z6vfqe3/files/media/images
              # 备注：需要visualization=dict(type='SegVisualizationHook', draw=True, interval=815))，draw=True才能看到特征图
              dict(type='WandbVisBackend')]

#####  配置6：configs/_base_/schedules/schedule_306k.py
更改 max_iters 和 val_interval、interval
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=306000, val_interval=204) 
logger=dict(type='LoggerHook', interval=204,

测试/验证期间结果可视化
特征图存储路径eg：work_dir/birenet/20230511_150800/vis_data/vis_image
wandb特征图存储路径eg：work_dir/birenet/20230511_150800/vis_data/wandb/run-20230511_150804-9z6vfqe3/files/media/images
visualization=dict(type='SegVisualizationHook', draw=True, interval=815))

#### 2、更改类别及后缀
mmseg/datasets/cityscapes.py
更改类别：
CLASSES = ('background','road')
PALETTE = [[0, 0, 0], [255, 255, 255]]
更改后缀：
img_suffix='.png',
seg_map_suffix='_road.png',


#### 3、常用指令
train
单gpu
python tools/train.py ${配置文件} --work-dir ${工作路径} --resume  load_from=${检查点}
load_from=${检查点} 指定检查点
--resume 默认从最新的检查点恢复
eg:
CUDA_VISIBLE_DEVICES=0  python tools/train.py  configs/birenet/birenet_1xb8-300epoch_deepglobe-1024x1024.py  --work-dir work_dir/birenet

多gpu
sh tools/dist_train.sh ${配置文件} ${GPU数量} --work-dir  ${工作路径} --resume  load_from=${检查点}
eg:
./tools/dist_train.sh  configs/birenet/birenet_1xb8-300epoch_deepglobe-1024x1024.py 2
--work-dir --work-dir work_dir/birenet

test
python tools/test.py ${配置文件} ${模型权重文件} [可选参数]
eg:
CUDA_VISIBLE_DEVICES=5 python tools/test.py configs/birenet/birenet_1xb8-300epoch_deepglobe-1024x1024.py  work_dir/birenet/best_mIoU_iter_XXX.pth

测试FPS
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs/birenet/birenet_1xb8-300epoch_deepglobe-1024x1024.py  work_dir/birenet/best_mIoU_iter_XXX.pth

测试FLOPs/Params
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/get_flops.py  configs/birenet/birenet_1xb8-300epoch_deepglobe-1024x1024.py   --shape 1024 1024

对 mIoU, mAcc, aAcc ,mFscore 指标画图
python tools/analysis_tools/analyze_logs.py work_dir/birenet/20240501_221552/vis_data/20240501_221552.json --keys mIoU mAcc aAcc mFscore --legend mIoU mAcc aAcc mFscore

对 loss 指标画图
python tools/analysis_tools/analyze_logs.py work_dir/birenet/20240508_195106/vis_data/20240508_195106.json --keys loss --legend loss


#### 4、测试/验证期间通过wandb实现特征图可视化
##### 配置1：configs/_base_/default_runtime.py
添加tensorboard和wandb
vis_backends=[dict(type='LocalVisBackend'),
              # tensorboard 路径eg:work_dir/birenet/20230510_200431/vis_data
              dict(type='TensorboardVisBackend'),
              # wandb 特征图路径eg：work_dir/birenet/20230511_150800/vis_data/wandb/run-20230511_150804-9z6vfqe3/files/media/images
              dict(type='WandbVisBackend')]

##### 配置2：configs/_base_/schedules/schedule_160k.py
测试/验证期间结果可视化
特征图存储路径eg：work_dir/birenet/20230511_150800/vis_data/vis_image
wandb特征图存储路径eg：work_dir/birenet/20230511_150800/vis_data/wandb/run-20230511_150804-9z6vfqe3/files/media/images
visualization=dict(type='SegVisualizationHook', draw=True, interval=815))


#### 5、通过脚本实现特征图可视化
脚本一：
tools/visualization/visualization.py
特征图存储路径eg：
tools/visualization/work_dirs/vis_data/vis_image/out_file_cityscapes_0.png

脚本二：
python tools/visualization/feature_map_visual.py \
tools/visualization/img/1.png \
configs/ann/ann_r50-d8_4xb2-40k_cityscapes-512x1024.py \
tools/visualization/test.pth \
--gt_mask tools/visualization/img/1_label.png
特征图存储路径（wandb）eg：
mmsegmentation/mmsegmentation/vis_data/wandb/run-20230511_151617-mb9txy83/files/media/images
