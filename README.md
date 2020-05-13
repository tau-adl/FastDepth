# FastDepth
# Requirements
Install PyTorch on a machine with a CUDA GPU. Our code was developed on a system running PyTorch v0.4.1.
Install the HDF5 format libraries. Files in our pre-processed datasets are in HDF5 format.
```
sudo apt-get update

sudo apt-get install -y libhdf5-serial-dev hdf5-tools

pip3 install h5py matplotlib imageio scikit-image opencv-python
```

Download the preprocessed NYU Depth V2 dataset in HDF5 format and place it under a data folder outside the repo directory. The NYU dataset requires 32G of storage space.
 ```
 mkdir data; cd data
 
 wget http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz
 
 tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz
 ```
