# PU-FPG: Point Cloud Upsampling via Form Preserving Graph Convolutional Networks

This is the official implementation for our paper PU-FPG: Point Cloud Upsampling via Form Preserving Graph Convolutional Networks. This repository supports training our PU-FPG, and previous methods [PU-Net](https://arxiv.org/abs/1801.06761), [MPU (3PU)](https://arxiv.org/abs/1811.11286), [PU-GAN](https://arxiv.org/abs/1907.10844) and [PU-GCN](https://arxiv.org/abs/1912.03264.pdf).

### Preparation

1. Clone the repository

   ```shell
   https://github.com/SATURN2021/PU-FPG.git
   cd PU-FPG
   ```

2. Install the environment step-by-step
   
   ```bash
   conda create -n pufpg python=3.6.8 cudatoolkit=10.0 cudnn numpy=1.16 -y
   conda activate pufpg
   pip install matplotlib tensorflow-gpu==1.13.1 open3d==0.9 Pillow gdown plyfile scikit-learn
   ```
   
3. Compilation for training
  
   ```bash
   cd tf_ops
   bash compile.sh linux
   cd .. 
   ```
   
   Please check `compile.sh` in `tf_ops` folder, one may have to manually change the path!!
   
4. Compilation for testing
   ```bash
   cd evaluation_code
   sudo apt-get install libcgal-dev -y
   bash compile.sh
   cd ..
   ```
5. Download PU1K dataset [Here](https://pan.baidu.com/s/1qKvkLeh4lxv4xhVboVVJYA) with the code `0gt3`, then unzip it

   ```bash
   unzip PU1K.zip
   ```

Our model is trained on a single NVIDIA Tesla V100-32G GPU and Intel Xeon E5-2698 v4 CPU.

### Train

Train models. We will provide our pretrained models soon.

-  PU-FPG
    ```shell
    python main.py --phase train --model pufpg 
    ```

-  PU-GCN
    ```shell
    python main.py --phase train --model pugcn --upsampler nodeshuffle --k 20 
    ```

-  PU-Net
    ```
    python main.py --phase train --model punet --upsampler original  
    ```

-  MPU
    ```
    python main.py --phase train --model mpu --upsampler duplicate 
    ```

-  PU-GAN
    ```
    python main.py --phase train --model pugan --more_up 2 
    ```



### Evaluation
The folder of models should be put under `/pretrain` folder and renamed as `pu1k-pufpg`
1. Test on PU1K dataset
   ```bash
   source test_pu1k.sh pretrain/pu1k-pufpg/ 0 0 --model pufpg
   ```

5. Test on Semantic3D dataset

    ```bash
    bash test_realscan.sh pretrain/pu1k-pufpg/  ./data/semantic3d 0 --model pufpg
    ```

### Acknowledgement
This repo is heavily built on [PU-GCN code](https://github.com/guochengqian/PU-GCN). Thanks to their splendid work. 


