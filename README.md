# Learning A Spiking Neural Network for Efficient Image Deraining
<!---
[![GoogleDrive](https://img.shields.io/badge/Data-GoogleDrive-brightgreen)](https://drive.google.com/drive/folders/1KRR_L276nviPT9JFPL9zfBiZVKJO6dM1?usp=drive_link)
[![BaiduPan](https://img.shields.io/badge/Data-BaiduPan-brightgreen)](https://pan.baidu.com/s/1TlgoslD-hIzySDL8l6gekw?pwd=pu2t)
--->

> **Abstract:** 
Recently, spiking neural networks (SNNs) have demonstrated substantial potential in computer vision tasks.
In this paper, we present an \textbf{E}fficient \textbf{S}piking \textbf{D}eraining \textbf{Net}work, called ESDNet.
Our work is motivated by the observation that rain pixel values will lead to a more pronounced intensity of spike signals in SNNs. However, directly applying deep SNNs to image deraining task still remains a significant challenge.
This is attributed to the information loss and training difficulties that arise from discrete binary activation and complex spatio-temporal dynamics.
To this end, we develop a spiking residual block to convert the input into spike signals, then adaptively optimize the membrane potential by introducing attention weights to adjust spike responses in a data-driven manner, alleviating information loss caused by discrete binary activation.
By this way, our ESDNet can effectively detect and analyze the characteristics of rain streaks by learning their fluctuations. This also enables better guidance for the deraining process and facilitates high-quality image reconstruction.
Instead of relying on the ANN-SNN conversion strategy, we introduce a gradient proxy strategy to directly train the model for overcoming the challenge of training. 
Experimental results show that our approach gains comparable performance against ANN-based methods while reducing energy consumption by 54\%. 

![RSDformer](figs/arch.png)

## News

- **July 4, 2023:** Paper submitted. 
- **Sep 13, 2023:** The basic version is released, including codes, pre-trained models on the Sate 1k dataset, and the used dataset.
- **Sep 14, 2023:** RICE dataset updated.
<!---  ** Sep 15, 2023:** The [visual results on Sate 1K](https://pan.baidu.com/s/1dToHnHI9GVaHQ3-I6OIbpA?pwd=rs1k) and [real-world dataset RSSD300](https://pan.baidu.com/s/1OZUWj8eo6EmP5Rh8DE1mrA?pwd=8ad5) are updated.-->


## Preparation

## Datasets
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Thin Haze</th>
    <th>Moderate Haze</th>
    <th>Thick Haze</th>
    <th>RICE</th>
    <th>RSSD300</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu Cloud</td>
    <td> <a href="https://pan.baidu.com/s/1r7RvVBKIj-viGxdxGE-HsQ?pwd=axjx ">Download</a> </td>
    <td align="center"> <a href="https://pan.baidu.com/s/1r7RvVBKIj-viGxdxGE-HsQ?pwd=axjx ">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1r7RvVBKIj-viGxdxGE-HsQ?pwd=axjx ">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1zbTBTys4VqL9CnJI0UFgoQ?pwd=7vj5">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1OZUWj8eo6EmP5Rh8DE1mrA?pwd=8ad5">Download</a> </td>
  </tr>
</tbody>
</table>
Here, the ''Thin haze'', ''Moderate haze'' and ''Thick haze'' are included in the Sate 1K dataset. We provide completely paired images, except for RRSD300, as it is a real-world remote sensing haze dataset. 

<!---
## Pre-trained Models
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Thin Haze</th>
    <th>Moderate Haze</th>
    <th>Thick Haze</th>
    <th>RICE</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu Cloud</td>
    <td> <a href="https://pan.baidu.com/s/1ncoc2qnlZd5hkSak6jEvrw?pwd=0nvo">Download</a> </td>
    <td align="center"> <a href="https://pan.baidu.com/s/1ncoc2qnlZd5hkSak6jEvrw?pwd=0nvo">Download</a> </td>
    <td > <a href="https://pan.baidu.com/s/1ncoc2qnlZd5hkSak6jEvrw?pwd=0nvo">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1OWtEGwccqzf6cmCtDhWZnA?pwd=gj56">Download</a> </td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td>Google Drive</td>
    <td> <a href="https://drive.google.com/drive/folders/1Dbja877w0TWXDqw9WYVwnRQMl6V9DkaT?usp=drive_link">Download</a> </td>
    <td align="center"> <a href="https://drive.google.com/drive/folders/1Dbja877w0TWXDqw9WYVwnRQMl6V9DkaT?usp=drive_link">Download</a> </td>
    <td> <a href="https://drive.google.com/drive/folders/1Dbja877w0TWXDqw9WYVwnRQMl6V9DkaT?usp=drive_link">Download</a> </td>
    <td> <a href="https://drive.google.com/drive/folders/1-d4OrxIbN3sN5coywpAvQavFIpQKxQwN?usp=drive_link">Download</a> </td>
  </tr>
</tbody>
</table>
Currently, we only provide the pre-trained models trained on the Sate 1K dataset. The pre-trained model of the RICE dataset will be updated as quickly as soon.
--->

### Install

We test the code on PyTorch 1.9.1 + CUDA 11.1 + cuDNN 8.0.5.

1. Create a new conda environment
```
conda create -n RSDformer python=3.8
conda activate RSDformer 
```

2. Install dependencies
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install matplotlib scikit-image opencv-python numpy einops math natsort tqdm lpips time tensorboardX
```

### Download

You can download the pre-trained models and datasets on Google Drive or BaiduPan.

Currently, we only provide the pre-trained model trained on the Sate 1K dataset and the used dataset (Sate 1K, RICE and RRSD300).  

The pre-trained models trained on RICE will be updated as quickly as possible.

The final file path should be the same as the following:

```
┬─ pretrained_models
│   ├─ thin_haze.pth
│   ├─ moderate_haze.pth
│   ├─ ... (model name)
│   └─ ... (exp name)
└─ data
    ├─ Sate_1K
    │├─ Haze1k_thick
    ││   ├─ train
    ││   │   ├─ input
    ││   │   │   └─ ... (image filename)
    ││   │   └─ target
    ││   │       └─ ... (corresponds to the former)
    ││   └─ test
    ││       └─ ...
    │└────  ... (dataset name)
    │
    │
    └─ ... (dataset name)

```
### Training, Testing and Evaluation

### Train
The training code will be released after the paper is accepted.
You should change the path to yours in the `Train.py` file.  Then run the following script to test the trained model:

```sh
python Train.py
```

### Test
You should change the path to yours in the `Test.py` file.  Then run the following script to test the trained model:

```sh
python Test.py
```

<!---
### Evaluation
You should change the path to yours in the `Dataload.py` file.  Then run the following script to test the trained model:

```sh
python PSNR_SSIM.py
```
It is recommended that you can download the visual deraining results and retest the quantitative results on your own device and environment.
--->

<!---
### Visual Results

<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Thin Haze</th>
    <th>Moderate Haze</th>
    <th>Thin Haze</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu Cloud</td>
    <td> <a href="https://pan.baidu.com/s/1dToHnHI9GVaHQ3-I6OIbpA?pwd=rs1k">Download</a> </td>
    <td align="center"> <a href="https://pan.baidu.com/s/1dToHnHI9GVaHQ3-I6OIbpA?pwd=rs1k">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1dToHnHI9GVaHQ3-I6OIbpA?pwd=rs1k">Download</a> </td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td>Google Drive</td>
    <td> <a href="https://drive.google.com/drive/folders/16UHn439SMJp0ZnDt_yoYc96ypsY7FN7n?usp=drive_link">Download</a> </td>
    <td align="center"> <a href="https://drive.google.com/drive/folders/16UHn439SMJp0ZnDt_yoYc96ypsY7FN7n?usp=drive_link">Download</a> </td>
    <td> <a href="https://drive.google.com/drive/folders/16UHn439SMJp0ZnDt_yoYc96ypsY7FN7n?usp=drive_link">Download</a> </td>
  </tr>
</tbody>
</table>
Currently, we provide the visual results on the Sate 1K dataset. The visual results of the RICE dataset and RSSD300 will be updated as quickly as soon.
--->

## Notes

1. Send e-mail to songtienyu@163.com if you have critical issues to be addressed.
2. Please note that there exists the slight gap in the final version due to errors caused by different testing devices and environments. 
3. Because the synthetic dataset is not realistic enough, the trained models may not work well on real hazy images.


## Acknowledgment

This code is based on the [Restormer](https://github.com/swz30/Restormer). The real-world dataset RRSD300 is collected from [RSHazeNet](https://github.com/chdwyb/RSHazeNet). Thanks for their awesome work.
