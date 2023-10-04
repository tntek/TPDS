
# TPDS

Code (pytorch) for ['Source-Free Domain Adaptation via Target Prediction Distribution Search']() on Digits(MNIST, USPS, SVHN), Office-31, Office-Home, VisDA-C, PACS. This paper has been accepted by International Journal of Computer Vision (IJCV). 
DOI: https://doi.org/10.1007/s11263-023-01892-w

### Preliminary

You need to download the [MNIST](http://yann.lecun.com/exdb/mnist/),[USPS](https://www.openml.org/search?type=data&sort=runs&id=41070&status=active),[SVHN](http://ufldl.stanford.edu/housenumbers/),[Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA ), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view),[PACS](https://github.com/MachineLearning2020/Homework3-PACS ), [VisDA-C](http://csr.bu.edu/ftp/visda17/clf/) dataset,  modify the path of images in each '.txt' under the folder './data/'.

The experiments are conducted on one GPU (NVIDIA RTX TITAN).

- python == 3.7.3
- pytorch ==1.6.0
- torchvision == 0.7.0


### Training and evaluation

Please refer to the file on [run.sh](./digit/run.sh).


### Citation
Tang, S., Chang, A., Zhang, F. et al. Source-Free Domain Adaptation via Target Prediction Distribution Searching. Int J Comput Vis (2023). https://doi.org/10.1007/s11263-023-01892-w

### Acknowledgement


The code is based on [DeepCluster(ECCV 2018)](https://github.com/facebookresearch/deepcluster) , [SHOT (ICML 2020, also source-free)](https://github.com/tim-learn/SHOT) and [IIC](https://github.com/sebastiani/IIC).


### Contact

- tntechlab@hotmail.com
