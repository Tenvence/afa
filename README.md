<div align="center">
<h1> Ensembling Diffusion Models via Adaptive Feature Aggregation</h1>

<p align="center">
Cong Wang<sup>1*</sup>, Kuan Tian<sup>2*</sup>, Yonghang Guan<sup>2</sup>, Jun Zhang<sup>2†</sup>, Zhiwei Jiang<sup>1†</sup>, Fei Shen<sup>2</sup>, Xiao Han<sup>2</sup>, Qing Gu<sup>1</sup>, Wei Yang<sup>2</sup>
<br>
<sup>1</sup> Nanjing University,
<sup>2</sup> Tencent AI Lab
<br>
<sup>*</sup> Equal contribution.
<sup>†</sup> Corresponding authors.
<br><br>
[<a href="https://arxiv.org/abs/2405.17082" target="_blank">arXiv</a>]
<br>
</div>

## Abstract

The success of the text-guided diffusion model has inspired the development and release of numerous powerful diffusion models within the open-source community.
These models are typically fine-tuned on various expert datasets, showcasing diverse denoising capabilities. 
Leveraging multiple high-quality models to produce stronger generation ability is valuable, but has not been extensively studied.
Existing methods primarily adopt parameter merging strategies to produce a new static model. 
However, they overlook the fact that the divergent denoising capabilities of the models may dynamically change across different states, such as when experiencing different prompts, initial noises, denoising steps, and spatial locations. 
In this paper, we propose a novel ensembling method, Adaptive Feature Aggregation (AFA), which dynamically adjusts the contributions of multiple models at the feature level according to various states (i.e., prompts, initial noises, denoising steps, and spatial locations), thereby keeping the advantages of multiple diffusion models, while suppressing their disadvantages.
Specifically, we design a lightweight Spatial-Aware Block-Wise (SABW) feature aggregator that adaptive aggregates the block-wise intermediate features from multiple U-Net denoisers into a unified one.
The core idea lies in dynamically producing an individual attention map for each model's features by comprehensively considering various states.
It is worth noting that only SABW is trainable with about 50 million parameters, while other models are frozen. 
Both the quantitative and qualitative experiments demonstrate the effectiveness of our proposed Adaptive Feature Aggregation method.

## Reference
```
@article{wang2024ensembling,
  title={Ensembling Diffusion Models via Adaptive Feature Aggregation},
  author={Wang, Cong and Tian, Kuan and Guan, Yonghang and Zhang, Jun and Jiang, Zhiwei and Shen, Fei and Han, Xiao and Gu, Qing and Yang, Wei},
  journal={arXiv preprint arXiv:2405.17082},
  year={2024}
}
```
