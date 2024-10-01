## 1. Train composed_image_retrieval

**change dir**
`cd composed_image_retrieval`

**train**
- 환경변수 설정
```
. env3/bin/activate
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
```

- 학습 실행

```
python -u src/main.py \
    --save-frequency 1 \
    --train-data="cc/Train_GCC-training_output.csv" #"cc/Train_LVIS_output.csv"  \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs=30 \
    --workers=8 \
    --openai-pretrained \
```

## 2. evaluate Region CLIP
**change dir**
`cd RegionCLIP`

**evaluate**
```
python3 ./tools/extract_region_features.py \
--config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \
INPUT_DIR ./datasets/custom_images \
OUTPUT_DIR ./output/region_feats \
```

## Thank you for Composed_image_retrieval, Region CLIP

```
@article{saito2023pic2word,
  title={Pic2Word: Mapping Pictures to Words for Zero-shot Composed Image Retrieval},
  author={Saito, Kuniaki and Sohn, Kihyuk and Zhang, Xiang and Li, Chun-Liang and Lee, Chen-Yu and Saenko, Kate and Pfister, Tomas},
  journal={CVPR},
  year={2023}
}
```
```
@inproceedings{zhong2022regionclip,
  title={Regionclip: Region-based language-image pretraining},
  author={Zhong, Yiwu and Yang, Jianwei and Zhang, Pengchuan and Li, Chunyuan and Codella, Noel and Li, Liunian Harold and Zhou, Luowei and Dai, Xiyang and Yuan, Lu and Li, Yin and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16793--16803},
  year={2022}
}
```
