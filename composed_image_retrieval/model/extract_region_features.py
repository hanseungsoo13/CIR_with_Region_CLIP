#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A script for region feature extraction
"""

import os
import torch
from torch.nn import functional as F
import numpy as np
import time

from detectron2.utils import comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
import pandas as pd
from PIL import Image, ImageDraw
from torch import nn
#from detectron2.modeling.meta_arch.clip_rcnn import visualize_proposals



class extract_region_feature(nn.Module):
    def __init__(self,input_resolution,args):
        super().__init__()
        self.args = args
        self.cfg = self.setup(self.args)
        self.model = self.create_model(self.cfg)
        self.input_resolution = input_resolution

    def setup(self,args):
        """
        Create configs and perform basic setups.
        """
        cfg = get_cfg()
        cfg.merge_from_file(args['config_file'])
        cfg.merge_from_list(args['opts'])
        cfg.freeze()
        default_setup(cfg, args)
        return cfg
    
    def create_model(self,cfg):
        """ Given a config file, create a detector
        (refer to tools/train_net.py)
        """
        # create model
        model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
        if cfg.MODEL.META_ARCHITECTURE in ['CLIPRCNN', 'CLIPFastRCNN', 'PretrainFastRCNN'] \
            and cfg.MODEL.CLIP.BB_RPN_WEIGHTS is not None\
            and cfg.MODEL.CLIP.CROP_REGION_TYPE == 'RPN': # load 2nd pretrained model
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, save_to_disk=True).resume_or_load(
                cfg.MODEL.CLIP.BB_RPN_WEIGHTS, resume=False
            )
        
        for p in model.parameters(): p.requires_grad = False
        model.eval()
        return model

    def get_inputs(self, cfg, images, texts):
        """
        Given a list of image file names and a list of texts, return a list of dictionaries.
        Each dict corresponds to a transformed image and its associated text.
        
        Args:
            cfg: Configuration object for the model.
            images (list): List of image file paths.
            texts (list): List of text descriptions corresponding to each image.
        
        Returns:
            List[dict]: List of dictionaries where each dictionary contains information for one image.
        """
        assert len(images) == len(texts), "Number of images and texts must be the same."
        
        # 리스트 형태로 입력받은 이미지와 텍스트를 하나씩 처리
        dataset_dicts = []
        for image_path, text in zip(images, texts):
            # 이미지 로드
            opened_image = Image.open(image_path)
            dataset_dict = {}

            # Detectron2의 utils.read_image 사용하여 이미지 읽기
            image = utils.read_image(image_path, format=cfg.INPUT.FORMAT)
            dataset_dict["height"], dataset_dict["width"] = image.shape[0], image.shape[1]  # h, w before transforms

            # 이미지 변환 (augmentation)
            augs = utils.build_augmentation(cfg, False)
            augmentations = T.AugmentationList(augs)  # [ResizeShortestEdge(short_edge_length=(800, 800), max_size=1333, sample_style='choice')]
            aug_input = T.AugInput(image)
            transforms = augmentations(aug_input)
            image = aug_input.image
            h, w = image.shape[:2]  # h, w after transforms

            # 변환된 이미지, 텍스트 및 기타 정보 저장
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            dataset_dict['text'] = text
            dataset_dict['original_image'] = opened_image

            # 변환된 결과를 리스트에 추가
            dataset_dicts.append(dataset_dict)

        return dataset_dicts    
    
    def forward(self,img,text,bbox=None, image_filenames=None):
        batched_inputs=self.get_inputs(self.cfg,image_filenames,text)
        
        """ Given a model and the input images, extract region features and save detection outputs into a local file
        (refer to detectron2/modeling/meta_arch/clip_rcnn.py)
        """
        # model inference  
        # 1. localization branch: offline modules to get the region proposals           
        images,texts, original_image = self.model.offline_preprocess_image(batched_inputs)
        if bbox == None:
            proposals = self.model.detector(original_image,texts) 
            #visualize_proposals(batched_inputs, proposals, model.input_format) 
            # 2. recognition branch: get 2D feature maps using the backbone of recognition branch
            images = self.model.preprocess_image(batched_inputs)
            features = self.model.backbone(images.tensor)

            # 3. given the proposals, crop region features from 2D image features
            proposal_boxes = [x['boxes'] for x in proposals]
            proposal_labels = [x['labels'] for x in proposals]
            proposal_scores = [x['scores'] for x in proposals]
            box_features = self.model.roi_heads._shared_roi_transform(
                [features[f] for f in self.model.roi_heads.in_features], proposal_boxes, self.model.backbone.layer4
            )
            att_feats = self.model.backbone.attnpool(box_features)  # region features

            if self.cfg.MODEL.CLIP.TEXT_EMB_PATH == 'None': # save features of RPN regions

                # save RPN outputs into files
                im_id = 0 # single image
                pred_boxes = proposal_boxes
                index = torch.argmax(proposal_scores)
                region_feats = att_feats[index] # region features, [#boxes, d]

                saved_dict = {}
                saved_dict['boxes'] = [i.cpu() for i in pred_boxes]
                saved_dict['classes'] = [i for i in proposal_labels]
                saved_dict['probs'] = [i.cpu() for i in proposal_scores]
                saved_dict['feats'] = att_feats.cpu()
            else: # save features of detection regions (after per-class NMS)
                # 4. prediction head classifies the regions (optional)
                predictions = self.model.roi_heads.box_predictor(att_feats)  # predictions[0]: class logits; predictions[1]: box delta
                pred_instances, keep_indices = self.model.roi_heads.box_predictor.inference(predictions, proposals) # apply per-class NMS
                results = self.model._postprocess(pred_instances, batched_inputs) # re-scale boxes back to original image size

                # save detection outputs into files
                im_id = 0 # single image
                pred_boxes = results[im_id]['instances'].get("pred_boxes").tensor # boxes after per-class NMS, [#boxes, 4]
                pred_classes = results[im_id]['instances'].get("pred_classes")# class predictions after per-class NMS, [#boxes], class value in [0, C]
                pred_probs = F.softmax(predictions[0], dim=-1)[keep_indices[im_id]] # class probabilities, [#boxes, #concepts+1], background is the index of C
                region_feats = att_feats[keep_indices[im_id]][0] # region features, [#boxes, d]
                # assert torch.all(results[0]['instances'].get("scores") == pred_probs[torch.arange(pred_probs.shape[0]).cuda(), pred_classes]) # scores

                saved_dict = {}
                saved_dict['boxes'] = pred_boxes.cpu()
                saved_dict['classes'] = pred_classes.cpu()
                saved_dict['probs'] = pred_probs.cpu()
                saved_dict['feats'] = region_feats.cpu()
        
        else:
            x, y, w, h = bbox
            x_min, y_min, x_max, y_max = x, y, x+w, y+h
            proposal_boxes = torch.stack([x_min, y_min, x_max, y_max]).T.to('cuda')
            proposal_boxes = [x for x in proposal_boxes]

            images = self.model.preprocess_image(batched_inputs)
            features = self.model.backbone(images.tensor)

            box_features = self.model.roi_heads._shared_roi_transform(
                [features[f] for f in self.model.roi_heads.in_features], proposal_boxes, self.model.backbone.layer4
            )
            att_feats = self.model.backbone.attnpool(box_features)
            region_feats = att_feats

        return region_feats
