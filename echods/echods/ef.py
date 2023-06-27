# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmaction.datasets.base import BaseDataset
from mmaction.datasets.builder import DATASETS
from mmaction.models.builder import LOSSES
from mmaction.models.builder import HEADS
import mmaction
from torch import nn
import torch
import numpy as np
from mmcv.utils import print_log
import torchmetrics
import echonet
import torchvision

@DATASETS.register_module()
class EchoDynamicDatasetEF(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self, ann_file, pipeline, start_index=0, **kwargs):
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        self.video_infos = []

        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(float, label))
                else:
                    filename, label = line_split
                    label = float(label)
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                self.video_infos.append(dict(filename=filename, label=label))
        return self.video_infos
    
    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                 logger=None,
                 **deprecated_kwargs):
        # print(f'\nresult@evaluate xx:{results}\n')
        R2Score = torchmetrics.R2Score()
        MeanAbsoluteError = torchmetrics.MeanAbsoluteError()
        MeanAbsolutePercentageError = torchmetrics.MeanAbsolutePercentageError()
        PearsonCorrCoef = torchmetrics.PearsonCorrCoef()
        SpearmanCorrCoef = torchmetrics.SpearmanCorrCoef()
        CosineSimilarityReductionMean = torchmetrics.CosineSimilarity(reduction = 'mean')
        # if logger != None
            # print_log(f'results:{results}, metrics:{metrics}', logger=logger)
        assert len(results) == len(self.video_infos), 'evaluate len does not match'
        x = []
        y = []
        for i in range(len(results)):
            x.append(results[i][0])
            y.append(self.video_infos[i]['label'])
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        return dict(
            R2Score=R2Score(x, y).item(),
            MeanAbsoluteError=MeanAbsoluteError(x, y).item(),
            MeanAbsolutePercentageError=MeanAbsolutePercentageError(x, y).item(),
            PearsonCorrCoef=PearsonCorrCoef(x, y).item(),
            SpearmanCorrCoef=SpearmanCorrCoef(x, y).item(),
            CosineSimilarityReductionMean=CosineSimilarityReductionMean(x, y).item(),
        )

@LOSSES.register_module()
class MyMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.loss = nn.MSELoss()
    def forward(self,
                input,
                label):
        # return self.loss(input.squeeze(), label)
        return torch.nn.functional.mse_loss(input.view(-1), label)

@DATASETS.register_module()
class EchoDynamicDatasetEF2(BaseDataset):
    """
        Args:
            limit, default -1, for limited dataset access, for testing purpose
            scale, default None, for video image desired output size. If size is a sequence like (h, w), output size will be matched to this. If size is an int, smaller edge of the image will be matched to this number. i.e, if height > width, then image will be rescaled to (size * height / width, size).
    """
    def __init__(self, split, data_dir, task='EF', frames=32, period=2, limit=-1, start_index=0, scale=None, **kwargs):
        super().__init__(start_index=start_index, **kwargs)
        self.split = split
         # Compute mean and std
        mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))
        self.limit = limit
        self.scale = scale
        self.efkwargs = {"target_type": task,
                "mean": mean,
                "std": std,
                "length": frames,
                "period": period,
                }

        # Set up datasets and dataloaders
        # dataset = {}
        self.inner_ds = echonet.datasets.Echo(root=data_dir, split=self.split, **self.efkwargs, pad=12)
    def __len__(self):
        if self.limit != -1: return self.limit
        return len(self.inner_ds)

    def __getitem__(self, idx):
        # c, f, h, w = video.shape
        # rgb imgs/np.array/float32
        video, ef = self.inner_ds[idx]
        if self.scale is not None:
            resizer = torchvision.transforms.Resize(self.scale)
            video = torch.from_numpy(video)
            video = resizer.forward(video)
            video = video.numpy()
        results = {}
        results['label'] = float(ef)
        results['img_norm_cfg'] = dict(
                mean=self.efkwargs['mean'], std=self.efkwargs['std'], to_bgr=False)
        results['num_clips'] = 1
        results['clip_len'] = self.efkwargs['length']
        # transform images into [M x H x W x C] format, to support mmaction pipelines, from `FormatShape`
        # M = 1 * N_crops * N_clips * L
        video = np.transpose(video, (1, 2, 3, 0))
        results['imgs'] = video
        results['original_shape'] = video[0].shape
        results['img_shape'] = video[0].shape
        # return by pipeline
        return self.pipeline(results)

    def load_annotations(self):
        pass

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                 logger=None,
                 **deprecated_kwargs):
        # print(f'\nresult@evaluate xx:{results}\n')
        R2Score = torchmetrics.R2Score(multioutput='raw_values')
        MeanAbsoluteError = torchmetrics.MeanAbsoluteError()
        MeanAbsolutePercentageError = torchmetrics.MeanAbsolutePercentageError()
        # SpearmanCorrCoef = torchmetrics.SpearmanCorrCoef()
        # CosineSimilarityReductionMean = torchmetrics.CosineSimilarity(reduction = 'mean')
        # if logger != None
            # print_log(f'results:{results}, metrics:{metrics}', logger=logger)
        # assert len(results) == len(self.video_infos), 'evaluate len does not match'
        x = []
        y = []
        for i in range(len(results)):
            x.append(results[i][0])
            y.append(self[i]['label'])
        x = torch.tensor(x)
        y = torch.tensor(y)
        return dict(
            R2Score=R2Score(x, y).item(),
            MeanAbsoluteError=MeanAbsoluteError(x, y).item(),
            MeanAbsolutePercentageError=MeanAbsolutePercentageError(x, y).item(),
            # SpearmanCorrCoef=SpearmanCorrCoef(x, y).item(),
            # CosineSimilarityReductionMean=CosineSimilarityReductionMean(x, y).item(),
        )

from mmaction.models.heads import BaseHead
from mmcv.cnn import normal_init, trunc_normal_init

@HEADS.register_module()
class MyI3DHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std, bias=55.6)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score

@HEADS.register_module()
class MyTimeSformerHead(BaseHead):
    """Classification head for TimeSformer.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to `dict(type='CrossEntropyLoss')`.
        init_std (float): Std value for Initiation. Defaults to 0.02.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 init_std=0.02,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.init_std = init_std
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        # self.softplus = nn.Softplus()

    def init_weights(self):
        """Initiate the parameters from scratch."""
        trunc_normal_init(self.fc_cls, std=self.init_std, bias=56.5)

    def forward(self, x):
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # cls_score = self.softplus(cls_score)
        # [N, num_classes]
        cls_score = cls_score
        return cls_score
