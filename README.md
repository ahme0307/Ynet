# Ynet
Y-Net: A deep Convolutional Neural Network to Polyp Detection

## Abstract
<p align="justify">
Colorectal polyps are important precursors to colon cancer, the third most common
cause of cancer mortality for both men and women. It is a disease where early detection
is of crucial importance. Colonoscopy is commonly used for early detection of cancer
and precancerous pathology. It is a demanding procedure requiring a significant amount
of time from specialized physicians and nurses, in addition to a significant miss-rates
of polyps by specialists. Automated polyp detection in colonoscopy videos has been
demonstrated to be a promising way to handle this problem. However, polyps detection
is a challenging problem due to the availability of limited amount of training data and
large appearance variations of polyps. To handle this problem, we propose a novel deep
learning method Y-Net that consists of two encoder networks with a decoder network.
Our proposed Y-Net method relies on efficient use of pre-trained and un-trained models
with novel sum-skip-concatenation operations. Each of the encoders are trained with
encoder specific learning rate along the decoder. Compared with the previous methods
employing hand-crafted features or 2-D/3-D convolutional neural network, our approach
outperforms state-of-the-art methods for polyp detection with 7.3% F1-score and 13%
recall improvement.</p>

## Sample Result
![alt text](https://github.com/ahme0307/Ynet/blob/master/readme/result.png)

## Network
![alt text](https://github.com/ahme0307/Ynet/blob/master/readme/network.png)


## How to train 
- To train the network, there is an interactive tool here <a href="https://github.com/ahme0307/Ynet/blob/master/YNet.ipynb">YNet.ipynb</a>   

Training data folder structure is 

- MainDir
  - Video1
    - Video1
      - Image0001.png
    - GT
      - Image0001_GT.png
  - Video2
    - Video2
      - Image0001.png
    - GT
      - Image0001_GT.png

## Testing
-  <a href="https://github.com/ahme0307/Ynet/blob/master/YNet.ipynb">YNet.ipynb</a>  

## Reference
If you find this code useful please cite

> Mohammed, Ahmed, et al. "Y-net: A deep convolutional neural network for polyp detection." arXiv preprint arXiv:1806.01907 (2018).

@article{mohammed2018net,
  title={Y-net: A deep convolutional neural network for polyp detection},
  author={Mohammed, Ahmed and Yildirim, Sule and Farup, Ivar and Pedersen, Marius and Hovde, {\O}istein},
  journal={arXiv preprint arXiv:1806.01907},
  year={2018}
}
