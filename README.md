# FastDepth Implemention in Pytorch
This repo contains Pytorch implementation of depth estimation deep learning network based on the published paper: [FastDepth: Fast Monocular Depth Estimation on Embedded Systems](https://arxiv.org/pdf/1903.03273.pdf)

This repository was part of the "Autonomous Robotics Lab" in Tel Aviv University

## Installation

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Requirements

This code was tested with:
* Ubuntu 18.04 with python 3.6.9

*The code also runs on Jetson TX2, for which all dependencies need to be installed via NVIDIA JetPack SDK.*


### Step-by-Step Procedure
In order to set the virtual environment, apriori installation of Anaconda3 platform is required.

Use the following commands to create a new working virtual environment with all the required dependencies.

**GPU based enviroment**:
```
git clone https://github.com/tau-adl/FastDepth
cd FastDepth
pip install -r pip_requirements.txt
```

# NYU database
Download the preprocessed NYU Depth V2 dataset in HDF5 format and place it under a data folder outside the repo directory. The NYU dataset requires 32G of storage space.
 ```
./DataCollect
```
# Train
```
python3 main.py -train -p 100 --epochs 20
```
# Evaluate
```
python3 main.py --evaluate /home/usr/results/trained_model.pth.tar
```
## Authors
* **Sunny Yehuda*** - *sunnyyehuda@gmail.com*
* **Gil Rafalovich*** - *rafalovichg@gmail.com*
* **David Sriker** - *David.Sriker@gmail.com
