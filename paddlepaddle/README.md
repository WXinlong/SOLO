# SOLO: Segmenting Objects by Locations - PaddlePaddle version

> This directory is a realization scheme of PaddlePaddle high-performance deep learning framework.

[PaddeDetection](https://github.com/PaddlePaddle/PaddleDetection) adopts the idea of modularization and has built-in rich data set enhancement strategy, backbone, network module and loss function. After understanding the excellent work of SOLO, we also implemented the PaddlePaddle version. We hope you can love it.

## Model Zoo

| Detector  | Backbone                | Multi-scale training  | Lr schd |  Mask AP<sup>val</sup> |  V100 FP32(FPS) |    GPU  |    Download                  | Configs |
| :-------: | :---------------------: | :-------------------: | :-----: | :--------------------: | :-------------: | :-----: | :---------: | :------------------------: |
| YOLACT++  |  R50-FPN    | False      |  80w iter     |   34.1 (test-dev) |  33.5  | Xp |  -  |  -   |
| CenterMask | R50-FPN | True        |   2x    |     36.4        |  13.9  | Xp |   -  |  -  |
| CenterMask | V2-99-FPN | True        |   3x    |     40.2       |  8.9  | Xp |   -  |  -  |
| PolarMask | R50-FPN | True        |   2x    |     30.5        |  9.4  | V100 |   -  |  -  |
| BlendMask | R50-FPN | True        |   3x    |     37.8        |  13.5  | V100 |   -  |  -  |
| SOLOv2 (Paper) | R50-FPN | False        |   1x    |     34.8        |  18.5  | V100 |   -  |  -  |
| SOLOv2 (Paper) | X101-DCN-FPN | True        |   3x    |     42.4        |  5.9  | V100 |   -  |  -  |
| SOLOv2 | R50-FPN                 |  False                |   1x    |    35.5         |  21.9     | V100 |  [model](https://paddledet.bj.bcebos.com/models/solov2_r50_fpn_1x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/solov2/solov2_r50_fpn_1x_coco.yml) |
| SOLOv2 | R50-FPN                 |  True                |   3x    |     38.0         |   21.9    | V100 |  [model](https://paddledet.bj.bcebos.com/models/solov2_r50_fpn_3x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/solov2/solov2_r50_fpn_3x_coco.yml) |
| SOLOv2 | R101vd-FPN                 |  True                |   3x    |     42.7         |   12.1    | V100 |  [model](https://paddledet.bj.bcebos.com/models/solov2_r101_vd_fpn_3x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/solov2/solov2_r101_vd_fpn_3x_coco.yml) |

**Notes:**

- SOLOv2 is trained on COCO train2017 dataset and evaluated on val2017 results of `mAP(IoU=0.5:0.95)`.

## Enhanced model
| Backbone                | Input size  | Lr schd | V100 FP32(FPS) | Mask AP<sup>val</sup> |         Download                  | Configs |
| :---------------------: | :-------------------: | :-----: | :------------: | :-----: | :---------: | :------------------------: |
| Light-R50-VD-DCN-FPN          |  512     |   3x    |     38.6          |  39.0   | [model](https://paddledet.bj.bcebos.com/models/solov2_r50_enhance_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/solov2/solov2_r50_enhance_coco.yml) |

**Optimizing method of enhanced model:**
- Better backbone network: ResNet50vd-DCN
- A better pre-training model for knowledge distillation
- [Exponential Moving Average](https://www.investopedia.com/terms/e/ema.asp)
- Synchronized Batch Normalization
- Multi-scale training
- More data augmentation methods
- DropBlock

## Training & Evaluation & Inference

PaddleDetection provides scripts for training, evalution and inference with various features according to different configure.    
[View full usage documentation](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.2/docs/tutorials/GETTING_STARTED_cn.md)

```bash
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/solov2/solov2_r50_fpn_3x_coco.yml
# training on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/solov2/solov2_r50_fpn_3x_coco.yml
# GPU evaluation
export CUDA_VISIBLE_DEVICES=0
python tools/eval.py -c configs/solov2/solov2_r50_fpn_3x_coco.yml -o weights=output/solov2_r50_fpn_3x_coco/model_final
# Inference
python tools/infer.py -c configs/solov2/solov2_r50_fpn_3x_coco.yml --infer_img=demo/000000570688.jpg -o weights=output/solov2_r50_fpn_3x_coco/model_final
```


## Citations
```
@article{wang2020solov2,
  title={SOLOv2: Dynamic and Fast Instance Segmentation},
  author={Wang, Xinlong and Zhang, Rufeng and  Kong, Tao and Li, Lei and Shen, Chunhua},
  journal={Proc. Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```


