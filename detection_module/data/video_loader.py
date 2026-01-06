import json
import random
# from datasets import load_dataset, Video

# import decord as de
import numpy as np
import torch
from operator import itemgetter

# from einops import rearrange
from skimage.feature import hog
from mmaction.apis import inference_recognizer, init_recognizer
import matplotlib.pyplot as plt


# from video_dataset import VideoFrameDataset

class DecordInit(object):
    """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1, **kwargs):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)
        self.kwargs = kwargs

    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(filename,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sr={self.sr},'
                    f'num_threads={self.num_threads})')
        return repr_str


if __name__ == '__main__':
    # Load the video
    img_path = "path_to_your_root/results/output_video.mp4"

    config_path = "configs/recognition/timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.py"

    checkpoint_path = 'https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_mit_rgb/tsn_r50_1x1x8_50e_hmdb51_mit_rgb_20201123-01526d41.pth'

    # build the model from a config file and a checkpoint file
    model = init_recognizer(config_path, checkpoint_path, device="cpu")  # device can be 'cuda:0'
    # test a single image
    results = inference_recognizer(model, img_path)
    # save the visualization results to a directory

    pred_scores = results.pred_score.tolist()
    score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
    top5_label = score_sorted[:5]



    label = 'tools/data/kinetics/label_map_k400.txt'
    labels = open(label).readlines()
    labels = [x.strip() for x in labels]
    results = [(labels[k[0]], k[1]) for k in top5_label]

    for result in results:
        print(f'{result[0]}: ', result[1])



