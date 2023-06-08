model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNeSt',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=30,
        in_channels=2048,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=30,
            reduction='mean',
            loss_weight=1.0),
        topk=(1, 5),
        cal_acc=False),
    train_cfg=dict(augments=dict(type='Mixup', alpha=0.2)))
dataset_type = 'My_ImageNet'
data_preprocessor = dict(
    num_classes=30,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandAugment',
        policies=[
            dict(type='AutoContrast', prob=0.5),
            dict(type='Equalize', prob=0.5),
            dict(type='Invert', prob=0.5),
            dict(
                type='Rotate',
                magnitude_key='angle',
                magnitude_range=(0, 30),
                pad_val=0,
                prob=0.5,
                random_negative_prob=0.5),
            dict(
                type='Posterize',
                magnitude_key='bits',
                magnitude_range=(0, 4),
                prob=0.5),
            dict(
                type='Solarize',
                magnitude_key='thr',
                magnitude_range=(0, 256),
                prob=0.5),
            dict(
                type='SolarizeAdd',
                magnitude_key='magnitude',
                magnitude_range=(0, 110),
                thr=128,
                prob=0.5),
            dict(
                type='ColorTransform',
                magnitude_key='magnitude',
                magnitude_range=(-0.9, 0.9),
                prob=0.5,
                random_negative_prob=0.0),
            dict(
                type='Contrast',
                magnitude_key='magnitude',
                magnitude_range=(-0.9, 0.9),
                prob=0.5,
                random_negative_prob=0.0),
            dict(
                type='Brightness',
                magnitude_key='magnitude',
                magnitude_range=(-0.9, 0.9),
                prob=0.5,
                random_negative_prob=0.0),
            dict(
                type='Sharpness',
                magnitude_key='magnitude',
                magnitude_range=(-0.9, 0.9),
                prob=0.5,
                random_negative_prob=0.0),
            dict(
                type='Shear',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.3),
                pad_val=0,
                prob=0.5,
                direction='horizontal',
                random_negative_prob=0.5),
            dict(
                type='Shear',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.3),
                pad_val=0,
                prob=0.5,
                direction='vertical',
                random_negative_prob=0.5),
            dict(
                type='Cutout',
                magnitude_key='shape',
                magnitude_range=(1, 41),
                pad_val=0,
                prob=0.5),
            dict(
                type='Translate',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.3),
                pad_val=0,
                prob=0.5,
                direction='horizontal',
                random_negative_prob=0.5,
                interpolation='bicubic'),
            dict(
                type='Translate',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.3),
                pad_val=0,
                prob=0.5,
                direction='vertical',
                random_negative_prob=0.5,
                interpolation='bicubic')
        ],
        num_policies=2,
        magnitude_level=12),
    dict(type='EfficientNetRandomCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(
        type='Lighting',
        eigval=[55.4625, 4.794, 1.1475],
        eigvec=[[-0.5836, -0.6948, 0.4203], [-0.5808, -0.0045, -0.814],
                [-0.5675, 0.7192, 0.4009]],
        alphastd=0.1,
        to_rgb=False),
    dict(type='PackInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetCenterCrop', crop_size=256, backend='pillow'),
    dict(type='PackInputs')
]
train_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type='My_ImageNet',
        data_root='data/imagenet',
        ann_file='train.txt',
        data_prefix='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='RandAugment',
                policies=[
                    dict(type='AutoContrast', prob=0.5),
                    dict(type='Equalize', prob=0.5),
                    dict(type='Invert', prob=0.5),
                    dict(
                        type='Rotate',
                        magnitude_key='angle',
                        magnitude_range=(0, 30),
                        pad_val=0,
                        prob=0.5,
                        random_negative_prob=0.5),
                    dict(
                        type='Posterize',
                        magnitude_key='bits',
                        magnitude_range=(0, 4),
                        prob=0.5),
                    dict(
                        type='Solarize',
                        magnitude_key='thr',
                        magnitude_range=(0, 256),
                        prob=0.5),
                    dict(
                        type='SolarizeAdd',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 110),
                        thr=128,
                        prob=0.5),
                    dict(
                        type='ColorTransform',
                        magnitude_key='magnitude',
                        magnitude_range=(-0.9, 0.9),
                        prob=0.5,
                        random_negative_prob=0.0),
                    dict(
                        type='Contrast',
                        magnitude_key='magnitude',
                        magnitude_range=(-0.9, 0.9),
                        prob=0.5,
                        random_negative_prob=0.0),
                    dict(
                        type='Brightness',
                        magnitude_key='magnitude',
                        magnitude_range=(-0.9, 0.9),
                        prob=0.5,
                        random_negative_prob=0.0),
                    dict(
                        type='Sharpness',
                        magnitude_key='magnitude',
                        magnitude_range=(-0.9, 0.9),
                        prob=0.5,
                        random_negative_prob=0.0),
                    dict(
                        type='Shear',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.3),
                        pad_val=0,
                        prob=0.5,
                        direction='horizontal',
                        random_negative_prob=0.5),
                    dict(
                        type='Shear',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.3),
                        pad_val=0,
                        prob=0.5,
                        direction='vertical',
                        random_negative_prob=0.5),
                    dict(
                        type='Cutout',
                        magnitude_key='shape',
                        magnitude_range=(1, 41),
                        pad_val=0,
                        prob=0.5),
                    dict(
                        type='Translate',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.3),
                        pad_val=0,
                        prob=0.5,
                        direction='horizontal',
                        random_negative_prob=0.5,
                        interpolation='bicubic'),
                    dict(
                        type='Translate',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.3),
                        pad_val=0,
                        prob=0.5,
                        direction='vertical',
                        random_negative_prob=0.5,
                        interpolation='bicubic')
                ],
                num_policies=2,
                magnitude_level=12),
            dict(type='EfficientNetRandomCrop', scale=224, backend='pillow'),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4),
            dict(
                type='Lighting',
                eigval=[55.4625, 4.794, 1.1475],
                eigvec=[[-0.5836, -0.6948, 0.4203], [-0.5808, -0.0045, -0.814],
                        [-0.5675, 0.7192, 0.4009]],
                alphastd=0.1,
                to_rgb=False),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type='My_ImageNet',
        data_root='data/imagenet',
        ann_file='val.txt',
        data_prefix='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='EfficientNetCenterCrop', crop_size=256,
                backend='pillow'),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type='My_ImageNet',
        data_root='data/imagenet',
        ann_file='val.txt',
        data_prefix='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='EfficientNetCenterCrop', crop_size=256,
                backend='pillow'),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
test_evaluator = dict(type='Accuracy', topk=(1, 5))
default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=50),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
policies = [
    dict(type='AutoContrast', prob=0.5),
    dict(type='Equalize', prob=0.5),
    dict(type='Invert', prob=0.5),
    dict(
        type='Rotate',
        magnitude_key='angle',
        magnitude_range=(0, 30),
        pad_val=0,
        prob=0.5,
        random_negative_prob=0.5),
    dict(
        type='Posterize',
        magnitude_key='bits',
        magnitude_range=(0, 4),
        prob=0.5),
    dict(
        type='Solarize',
        magnitude_key='thr',
        magnitude_range=(0, 256),
        prob=0.5),
    dict(
        type='SolarizeAdd',
        magnitude_key='magnitude',
        magnitude_range=(0, 110),
        thr=128,
        prob=0.5),
    dict(
        type='ColorTransform',
        magnitude_key='magnitude',
        magnitude_range=(-0.9, 0.9),
        prob=0.5,
        random_negative_prob=0.0),
    dict(
        type='Contrast',
        magnitude_key='magnitude',
        magnitude_range=(-0.9, 0.9),
        prob=0.5,
        random_negative_prob=0.0),
    dict(
        type='Brightness',
        magnitude_key='magnitude',
        magnitude_range=(-0.9, 0.9),
        prob=0.5,
        random_negative_prob=0.0),
    dict(
        type='Sharpness',
        magnitude_key='magnitude',
        magnitude_range=(-0.9, 0.9),
        prob=0.5,
        random_negative_prob=0.0),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=0,
        prob=0.5,
        direction='horizontal',
        random_negative_prob=0.5),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=0,
        prob=0.5,
        direction='vertical',
        random_negative_prob=0.5),
    dict(
        type='Cutout',
        magnitude_key='shape',
        magnitude_range=(1, 41),
        pad_val=0,
        prob=0.5),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=0,
        prob=0.5,
        direction='horizontal',
        random_negative_prob=0.5,
        interpolation='bicubic'),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=0,
        prob=0.5,
        direction='vertical',
        random_negative_prob=0.5,
        interpolation='bicubic')
]
EIGVAL = [55.4625, 4.794, 1.1475]
EIGVEC = [[-0.5836, -0.6948, 0.4203], [-0.5808, -0.0045, -0.814],
          [-0.5675, 0.7192, 0.4009]]
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.8, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-06,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=265, by_epoch=True, begin=5, end=270)
]
train_cfg = dict(by_epoch=True, max_epochs=270)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=2048)
launcher = 'none'
work_dir = './work_dirs/my_resnest50_32xb64_in1k'
