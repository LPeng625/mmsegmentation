_base_ = [
    '../_base_/models/birenet.py',
    '../_base_/datasets/deepglobe.py',
    '../_base_/default_runtime.py'
]
import math

# TODO config setting
train_dataset_size = 5500
val_dataset_size = 726
train_batch_size = 2
val_batch_size = 1
warmup_epoch = 5
epoch = 300
LinearLR_iters = math.ceil(train_dataset_size * warmup_epoch / train_batch_size)
total_iters = math.ceil(train_dataset_size * epoch / train_batch_size)
step_size = math.ceil(train_dataset_size * 15 / train_batch_size)
milestones = [step_size, math.ceil(step_size * 1.1), math.ceil(step_size * 1.15), math.ceil(step_size * 1.2)]
crop_size = (1024, 1024)  # TODO 为了加速训练，裁剪图片大小为512x512

optimizer = dict(type='AdamW', lr=2 * 1e-4, eps=1e-08, weight_decay=0)
# optimizer = dict(type='AdamW', lr=train_batch_size * 1e-4, eps=1e-08, weight_decay=0)


optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
# learning policy
param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=LinearLR_iters),
    # dict(
    #     type='PolyLR',
    #     eta_min=1e-6,
    #     power=0.9,
    #     by_epoch=False,
    #     begin=0,
    #     end=total_iters)
    # dict(
    #     type='MultiStepLR',
    #     begin=LinearLR_iters,
    #     end=total_iters,
    #     milestones=milestones,
    #     gamma=0.2,
    #     by_epoch=False)
    # dict(
    #     type='StepLR',
    #     begin=LinearLR_iters,
    #     end=total_iters,
    #     step_size=step_size,
    #     gamma=0.9,
    #     by_epoch=False)
    # dict(
    #     type='ExponentialLR',
    #     begin=LinearLR_iters,
    #     end=total_iters,
    #     gamma=0.99999,
    #     by_epoch=False)
    dict(
        type='CosineAnnealingLR',
        T_max=total_iters - LinearLR_iters,
        by_epoch=False,
        begin=LinearLR_iters,
        end=total_iters,
        eta_min=train_batch_size * 1e-6),
    # dict(
    #     eta_max=train_batch_size * 1e-4,
    #     three_phase=False,
    #     type='OneCycleLR',
    #     by_epoch=False,
    #     total_steps=total_iters-LinearLR_iters
    #     ),
]

# custom_hooks = [
#     dict(
#         type='CustomLRSchedulerHook',
#         patience=3,
#         factor=0.2,
#         min_lr=5e-7,
#         cooldown=6,
#         priority='ABOVE_NORMAL'
#     )
# ]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=total_iters, val_interval=val_dataset_size)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=val_dataset_size, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=val_dataset_size, save_best='mIoU',
                    max_keep_ckpts=5),
    # TODO 保存mIoU最高的模型, 限制保存checkpoint数量
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=train_batch_size, num_workers=4, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=val_batch_size, num_workers=4, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# TODO
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'],
                     output_dir='work_dir/birenet/format_results')
test_evaluator = val_evaluator
