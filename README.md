# TPAMI 2025 Spatial Frequency Modulation for Semantic Segmentation

This repository contains implementation for the paper: **Spatial Frequency Modulation for Semantic Segmentation** (Accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence, TPAMI 2025).

![image-20250715191310237](README.assets/image-20250715191310237.png)

In this work, we identify and address the "aliasing degradation" problem in modern deep neural networks, where high-frequency information crucial for semantic segmentation is distorted during downsampling.

Instead of simply filtering out these valuable details, we introduce **Spatial Frequency Modulation (SFM)**, a novel framework that:

1. **Modulates** high-frequency features to a lower frequency band *before* downsampling, protecting them from aliasing.
2. **Demodulates** these features back to their original high frequency during upsampling, recovering fine-grained details for a more accurate segmentation.

Our lightweight and plug-and-play modules, **Adaptive Resampling (ARS)** and **Multi-Scale Adaptive Upsampling (MSAU)**, can be seamlessly integrated into various CNN and Transformer architectures to significantly boost their performance.

![image-20250715191346558](README.assets/image-20250715191346558.png)

> **Figure:** An illustration of our SFM framework. Adaptive Resampling (ARS) is inserted before downsampling layers to perform frequency modulation, and Multi-Scale Adaptive Upsampling (MSAU) is used to demodulate the features and produce the final high-resolution segmentation map.

## Main Contributions

- **Identifying "Aliasing Degradation":** We quantitatively demonstrate that a higher aliasing ratio in feature maps leads to lower segmentation accuracy, providing a clear motivation for frequency-aware network design.
- **Spatial Frequency Modulation (SFM):** We propose a novel framework to preserve high-frequency details by modulating them to lower frequencies to survive downsampling, and then demodulating them to recover the details.
- **Lightweight and Effective Modules:** We implement SFM with two novel modules:
  - **Adaptive Resampling (ARS):** A lightweight module that learns to densely sample high-frequency regions (e.g., boundaries, textures) to effectively lower their frequency representation.
  - **Multi-Scale Adaptive Upsampling (MSAU):** A module that performs non-uniform upsampling to reverse the modulation and refines segmentation details by modeling multi-scale pixel relationships.
- **Broad Applicability:** Our method is model-agnostic and consistently improves various state-of-the-art segmentation models, including CNN-based (ResNet, ConvNeXt, InternImage) and Transformer-based (Swin) architectures, with minimal computational overhead.

## Model

| Model                                                        | Dataset | mIoU                         |
| ------------------------------------------------------------ | ------- | ---------------------------- |
| Mask2Former ([code](./SFM_mask2former/), [config](./SFM_mask2former/configs/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512_SFM.py), [ckpt](https://pan.baidu.com/s/1s9LlTZgA9s7kV_8xXDGZVA?pwd=PAMI)) | ADE20K  | 47.7 $\rightarrow$  **49.2** |
|                                                              |         |                              |
|                                                              |         |                              |



## Citation

If you find our work useful in your research, please consider citing our paper:

Generated bibtex

```
@article{chen2023spatialfrequency,
  title={Spatial Frequency Modulation for Semantic Segmentation},
  author={Chen, Linwei and Fu, Ying and Gu, Lin and Zheng, Dezhi and Dai, Jifeng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}
    
```

## Acknowledgements

This project is built upon the excellent [MMSegmentation](https://www.google.com/url?sa=E&q=https%3A%2F%2Fgithub.com%2Fopen-mmlab%2Fmmsegmentation) toolbox. We thank the authors for their open-source contribution.

## Contact

If you encounter any problems or bugs, please don't hesitate to contact me at [chenlinwei@bit.edu.cn](chenlinwei@bit.edu.cn), [charleschen2013@163.com](charleschen2013@163.com). To ensure effective assistance, please provide a brief self-introduction, including your name, affiliation, and position. If you would like more in-depth help, feel free to provide additional information such as your personal website link. I would be happy to discuss with you and offer support.

