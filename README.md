# Inevitable Trade-off between Watermark Strength and Speculative Sampling Efficiency for Language Models

This repository contains the code for the paper "Inevitable Trade-off between Watermark Strength and Speculative Sampling Efficiency for Language Models" by Zhengmian Hu and Heng Huang, presented at the Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024).

## Introduction

Watermarking techniques help identify AI-generated content, but they need to be accelerated to become practical. This project explores the combination and trade-off between watermark strength and speculative sampling efficiency, in the context of accelerating the generation of watermarked tokens for large language models.

![illustration](https://github.com/user-attachments/assets/1a6a33a3-5ed0-41cf-817e-d8beabf6c95a)

- We propose a two-reweight framework that allows for the integration of unbiased watermarking and speculative sampling techniques while preserving the output distribution.
- We prove a no-go theorem, demonstrating that it is impossible to simultaneously maintain the highest watermark strength and the highest sampling efficiency within the two-reweight framework when the vocabulary size is greater than 2.
- We present two practical algorithms that prioritize either watermark strength or sampling efficiency.


## Repository Structure

- `unbiased_watermark/`: Implements unbiased reweighting functions that preserve output quality.
- `accuwm/`: Contains five language model inference algorithms:
  - No watermark, no acceleration
  - Watermark, no acceleration
  - No watermark, acceleration
  - Acceleration while maintaining watermark strength
  - Watermarking while maintaining speculative sampling efficiency
- `experiments/`: Includes experiments from the paper, requiring approximately 1200 A6000 GPU hours.
- `analysis/`: Aggregates experimental results into figures and tables presented in the paper.

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{
  hu2024inevitable,
  title={Inevitable Trade-off between Watermark Strength and Speculative Sampling Efficiency for Language Models},
  author={Hu, Zhengmian and Huang, Heng},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
}
```
