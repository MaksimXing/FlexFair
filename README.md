# FlexFair

FlexFair: Achieving Flexible Fairness Metrics in Federated Learning for Medical Image Analysis

## Clone repository

```bash
git clone https://github.com/MaksimXing/FlexFair.git
```

## Download dataset

### Ployp Dataset

- The training and testing datasets come from [PraNet](https://github.com/DengPingFan/PraNet). Download these datasets and unzip them into `data` folder

  - [Training Dataset](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view?usp=sharing)
  - [Testing Dataset](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view?usp=sharing)

However, We modified the dataset path to match the purpose of federated learning training, and the files can be accessed by going to [Google Drive](https://drive.google.com/drive/folders/1RXtECSl5GoULQD_WE0cecGo1KrtBwx6x?usp=sharing) to obtain the dataset we used by downloading the file with the name `Public_5.zip`.

### Fundus Vascular Dataset

The datasets include CHASE\_DB1, DRIVE, and STARE, accessible at [CHASE_DB1](https://www.kaggle.com/datasets/khoongweihao/chasedb1), [DRIVE](https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction), and [STARE](https://paperswithcode.com/dataset/stare), respectively.

The files can also be accessed by going to [Google Drive](https://drive.google.com/drive/folders/1RXtECSl5GoULQD_WE0cecGo1KrtBwx6x?usp=sharing) to obtain the dataset we used by downloading the file with the name `fundus_data.zip`.

### Skin Disease Dataset

The original image data can be obtained from [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and [BCN20000](https://paperswithcode.com/dataset/bcn-20000).

However, we used data sieved free of age and gender information for fairness training, and the files can be accessed by going to [Google Drive](https://drive.google.com/drive/folders/1RXtECSl5GoULQD_WE0cecGo1KrtBwx6x?usp=sharing) to obtain the dataset we used by downloading the file with the name `SkinDisease.zip`.

### Private Cervical Cancer Dataset

In line with Nature’s open-data principles and to ensure permanent linkage, we have deposited the dataset on Figshare (DOI: 10.6084/m9.figshare.28636133), files can be accessed by going to [Figshare](https://doi.org/10.6084/m9.figshare.28639598). 
The files can be accessed by going to [Google Drive](https://drive.google.com/drive/folders/1RXtECSl5GoULQD_WE0cecGo1KrtBwx6x?usp=sharing) to obtain the dataset we used by downloading the file with the names `train.zip` and `test.zip`.

## Prerequisites

Please use the following command to install the dependencies:

```bash
pip install -r requirements.txt
```







## Figure redrawing on the paper

Access all the code used for plotting in the paper by executing the following command:

```bash
cd Figure
```

Please note that some code implementations need to specify the dataset path for analysis, please refer to the internal comments and readme.txt for details

## Paper results reproduced

We offer two ways to reproduce the results:

1. By running the training process data we provide (pre-training data) to redraw the image
2. Reproduce the results by fixing the seed and weight to start the training process all over again

Due to data size limitations, the pre-training data requires access to [Google Drive](https://drive.google.com/drive/folders/1RXtECSl5GoULQD_WE0cecGo1KrtBwx6x?usp=sharing) to obtain.

Detailed run methods for each dataset are stored in the readme of each folder under `Code`, please refer to it for a run guide.

