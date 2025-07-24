# 7-Type Classification GeoProj - Part of the Deep-Learning Model for Mirror Detection FYP

### [Paper of the Original Project](https://arxiv.org/abs/1909.03459)

The source code of 7-Type Classification GeoNet of Deep-Learning Model for Mirror Detection FYP (2025). 
The ClassNet has been modified for adding a new 'None' Type for identifying distorted and non-distorted images.

Changes include:
- data/dataset_generate.py
- data/distortion_model.py
- data/visualise_flow.py
- dataloaderNetM.py
- eval.py
- modelNetM.py
- train.py

## Prerequisites
- Linux or Windows
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Dataset Generation
In order to train the model using the provided code, the data needs to be generated in a certain manner. 

You can use any distortion-free images to generate the dataset. In this paper, we use [Places365-Validation dataset](http://places2.csail.mit.edu/download.html) at the resolution of 512\*512 as the original non-distorted images to generate the 256\*256 dataset.

Run the following command for dataset generation:
```bash
python data/dataset_generate.py [--sourcedir [PATH]] [--datasetdir [PATH]] [--cleardatasetdir [BOOLEAN]]
condor_submit [condor_script] # if you want to submit the task on HTCondor

--sourcedir           Path to original non-distorted images
--datasetdir          Path to the generated dataset
--cleardatasetdir     Clears existing dataset directory automatically if True. Default=False
```

### Training
Run the following command for help message about optional arguments like learning rate, dataset directory, etc.
```bash
python trainNetM.py --h # if you want to train GeoNetM
condor_submit [condor_script] # if you want to submit the task on HTCondor
```

### Use a Pre-trained Model
You can download the pretrained models [here](https://drive.google.com/drive/folders/1k6kLYyvqaUST3m-odXj-Lxr6YPJ4TFo_?usp=sharing).

You can also use `eval.py` and modify the model path, image path and saved result path to your own directory to generate your own results.

### Resampling ()
Import `resample.resampling.rectification` function to resample the distorted image by the forward flow.

The distorted image should be a Numpy array with the shape of H\*W\*3 for a color image or H\*W for a greyscale image, the forward flow should be an array with the shape of 2\*H\*W.

The function will return the resulting image and a mask to indicate whether each pixel will converge within the maximum iteration.

## Citation
Please refer to the original project repository.
