norm_cfg = dict(type='BN', eps=1e-03, requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[73.4950, 97.6461, 104.5076],
    std=[31.2847, 32.3201, 39.8365],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='BiReNet34',
        out_channels=2),
    decode_head=dict(
        type='BireNetHead',
        out_channels=2,
        in_channels=64,
        in_index=0,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(
                type='DiceLoss', use_sigmoid=False, loss_weight=1.0)
        ]),
    auxiliary_head=dict(
        type='BireNetAnxiliaryHead',
        out_channels=2,  # TODO sigmoid = 1, softmax =2
        in_channels=64,
        in_index=1,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0,
        num_classes=2,  # 实际类别
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(
                type='DiceLoss', use_sigmoid=False, loss_weight=1.0)
        ]),
    # loss_decode=[
    #     # CrossEntropyLoss的变体，通过gamma和alpha调整正负样本的权重，
    #     # 即相比交叉熵损失，focal loss对于分类不准确的样本，损失几乎没有改变，对于分类准确的样本，损失会变小
    #     # 主要是gamma起最大的作用
    #     dict(
    #         type='FocalLoss',  # TODO 只能sigmoid,softmax的方法未实现 理论上out_channels为1和2均可
    #         gamma=2.0,  # gamma=2.0
    #         alpha=0.5,  # alpha=0.5
    #         loss_weight=1.0),  # 1.0
    #     dict(
    #         # dice的变体，通过alpha和beta去控制召回率和精确度，alpha准确率权重，beta召回率权重，默认alpha=0.3,beta=0.7
    #         # 使用TverskyLoss，out_channels必须大于1
    #         type='TverskyLoss',  # TODO 只能softmax 内部添加sigmoid精度更高
    #         alpha=0.3,
    #         beta=0.7,
    #         loss_weight=1.0),
    # ]),
    train_cfg=dict(sampler=None),
    test_cfg=dict(mode='whole'))
