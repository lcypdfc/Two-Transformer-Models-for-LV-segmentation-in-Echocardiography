_base_ = [
    '../mmsegmentation/configs/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py'
]

norm_cfg=dict(type='BN', requires_grad=True)

model = dict(
    decode_head=dict(num_classes=2, norm_cfg=norm_cfg)
)

custom_imports = dict(imports=['echods.seg'], allow_failed_imports=False)

train_pipeline=[
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline=[
    dict(type='MultiScaleFlipAug',
        img_scale=(224, 224),
        img_ratios=None,
        flip=False,
        transforms=[
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
]

data_root = '/data/EchoNet-Dynamic'
data = dict(
    samples_per_gpu=40,
    workers_per_gpu=4,
    train=dict(
        type='EchoDynamicDatasetSeg',
        data_root=data_root,
        split='train',
        pipeline=train_pipeline
    ),
    val=dict(
        type='EchoDynamicDatasetSeg',
        data_root=data_root,
        split='val',
        pipeline=test_pipeline
    ),
    test=dict(
        type='EchoDynamicDatasetSeg',
        data_root=data_root,
        split='test',
        pipeline=test_pipeline
    ),
)

auto_resume=True # useless, must be set from cmdline
evaluation = dict(by_epoch=True, interval=5, metric=['mIoU','mDice'], pre_eval=True)
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(by_epoch=True, interval=1)
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=True)])
