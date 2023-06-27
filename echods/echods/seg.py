from cgitb import reset
from re import X
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

import echonet
from echonet.datasets import Echo 
import numpy as np
import math

from mmseg.datasets.pipelines.compose import Compose


@DATASETS.register_module()
class EchoDynamicDatasetSeg(CustomDataset):
  CLASSES = ('ventricle', 'backgound')
  PALETTE = [[255, 255, 255], [0, 0, 0]]
  echods:Echo = None
  tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace", "Filename"]

  def __init__(self, data_root, split, pipeline=[], test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False, **kwargs):
    # super().__init__(img_suffix='.jpg', seg_map_suffix='.png',
                    #  split=split, **kwargs)
    self.test_mode = test_mode
    self.ignore_index = ignore_index
    self.reduce_zero_label = reduce_zero_label
    self.label_map = None
    self.pipeline = Compose(pipeline)
    self.data_root = data_root
    self.split = split
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=self.data_root, split=self.split))
    args = {"target_type": self.tasks,
            "mean": mean,
            "std": std
            }
    self.echods = Echo(root=self.data_root, split=self.split, **args)
  
  def prepare_by_pipeline(self, results):
    return self.pipeline(results)

  def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
    pass
  
  def __len__(self):
    return len(self.echods) * 2
  
  def __getitem__(self, idx):
    results = self._get_sample(idx)
    return self.prepare_by_pipeline(results)

  def _get_sample(self, idx):
    n = math.floor(idx / 2)
    (_, sample) = self.echods[n]
    xlarge,xsmall,ylarge,ysmall, filename = sample
    xlarge = xlarge.transpose(1,2,0)
    xsmall = xsmall.transpose(1,2,0)
    # ylarge = ylarge.astype(np.float)
    # ysmall = ysmall.astype(np.float)
    results = {}
    if idx > 2*n : # odd
      results['img'] = xsmall
      results['gt_semantic_seg'] = ysmall
    else:
      results['img'] = xlarge
      results['gt_semantic_seg'] = ylarge
    results['img_info'] = dict()
    results['ann_info'] = dict()
    results['seg_fields'] = ['gt_semantic_seg']
    results['filename'] = filename + f"_{idx}.png"
    results['ori_filename'] = filename + f"_{idx}.png"
    results['img_shape'] = xlarge.shape
    results['ori_shape'] = xlarge.shape
    results['pad_shape'] = (0, 0)
    results['scale_factor'] = (1, 1)
    results['flip'] = False
    results['flip_direction'] = 'horizontal'
    results['img_norm_cfg'] = dict()
    return results

  def get_gt_seg_map_by_idx(self, index):
    results = self._get_sample(index)
    return results['gt_semantic_seg']

  def get_gt_seg_maps(self, efficient_test=None):
    for idx in range(len(self)):
      yield self.get_gt_seg_map_by_idx(idx)

  def get_ann_info(self, idx):
    return dict(seg_map="not-exist-file-path")