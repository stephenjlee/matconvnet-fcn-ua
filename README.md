# MIT-Comillas Universal Access Fork of MatConvNet FCN

## Introduction
We have modified files from the original MatConvNet FCN branch in this fork. We are specifically using the computer vision system to extract buildings from satellite imagery.

## Installing in Linux (tested on Ubuntu 14.04)
* Install MATLAB with GUI
* Install Cuda (using Deb installer)

* Revert to gcc 4.7

```
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev protobuf-compiler gcc-4.7 g++-4.7 gcc-4.7-multilib g++-4.7-multilib gfortran libjpeg62 libfreeimage-dev libatlas-base-dev git python-dev python-pip

sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-4.7 60
sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-4.7 60
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.7 60
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.7 60

echo 'export CXX=/usr/bin/gcc-4.7' >> ~/.bashrc
```

* Download and unzip matconvnet-fcn-master-ua.zip, go into the matlab folder, and ensure that matconvnet is setup properly for your machine. One way to do this is to try running the peppers [demo] (http://www.vlfeat.org/matconvnet/quick/).
* To enable GPU support, go to the matconvnet/matlab directory and run `vl_compilenn('enableGpu', true)`


## Directory Structure
* **[data]**
  * **[fcn8s-satellite_XXX]** - where the **imdb.mat**, **imdbStats.mat**, **net-train.pdf**, and **net-epoch-###.mat** files will be saved after running **fcnTrain_satellite.m**
    * **imdb.mat** - contains links to all of the training data and test data. It gets defined via the evaluating the data in the data/satellite/ImageSets folder
    * **imdbStats.mat** - contains descriptive data about the image sets
    * **net-train.pdf** - a pdf of the learning curves. It shows increasing accuracy and decreasing objective functions for both the training and validation sets
    * **net-epoch-###.mat** - the neural network itself
  * **[satellite]**
    * **[ImageSets]**
      * **[Main]**
        * **building_train.txt** - a list of all of the images in the training set that have buildings. One line per image; no file extension needed. After the image name, there's a space, followed by 1 or -1. Add a -1 if there is no building, and a 1 if there is a building.
        * **building_trainval.txt** - a combined list of all of the images in the training set and in the validation set. One line per image; no file extension needed. After the image name, there's a space, followed by 1 or -1. Add a -1 if there is no building, and a 1 if there is a building.
        * **building_val.txt** - a list of all of the images in the validation set. One line per image; no file extension needed. After the image name, there's a space, followed by 1 or -1. Add a -1 if there is no building, and a 1 if there is a building.
      * **[Segmentation]**
        * **train.txt** - a list of all of the images in the training set. One line per image; no file extension needed.
        * **trainval.txt** - a combined list of all of the images in the training set and in the validation set. One line per image; no file extension needed.
        * **val.txt** - a list of all of the images in the validation set. One line per image; no file extension needed.
    * **[JPEGImages]** -- put all of the jpg training data here cooresponding to the ground truth
    * **[SegmentationClassExt]** -- put all png ground truth data here. All of these pngs should correspond to the VOClabelcolormap
  * **[models]** - this is where the models should go that you are fine-tuning from
    * **imagenet-vgg-verydeep-16.mat** - the base model we have used for find-tuning
  * **[modelzoo]**
    * **[[folderName]]** - contains the inferences from the various runs
      * **imdb.mat** - required to be copied from data/fcn8s-satellite_XXX for the inferences to run
* **fcnTrain_satellite.m** - fine-tuning neural networks
* **fcnTest_satellite.m** - testing neural networks to evaluate error metrics
* **fcnRun_satellite.m** - using neural networks to produce inferences

## Fine-tuning

* Set up all of the txt files in data/satellite/ImageSets to their correct values as defined above. Use the matlab function in the universal-access-geoprocessing project called **detectBuildingsInSegmentations.m** and the excel file in data/satellite called **building_train.xlsx**  to help with this
* Ensure that the correct training images are in **data/satellite/JPEGImages** and **data/satellite/SegmentationClassExt**
* Configure **fcnTrain_satellite.m** to point to the correct files and folders
  * *opts.expDir* - points to a new **data/fcn8s-satellite_XXX** folder
  * *opts.dataDir* - points to the right **data/satellite** folder that holds the training and validation sets
  * *otps.sourceModelPath* - points to the correct base model for find-tuning from. For us, this is usually **data/models/imagenet-vgg-verdeep-16.mat**
  * *opts.train.gpus* - this this equal to [1] if using one GPU. Set this equal to the number of GPUs you want to find-tune on.
  * *opts.train.learningRate = 0.0001 * ones(1,500)* - Set the second number in the ones function to the number of epochs you want the system to find-tune. In this case, we are find-tuning for 500 epochs. One neural net will be saved for each epoch.

## Testing

* Set up **fcnTest_satellite.m** much the same as above; however, now we want to be pointing to a few new places, since we are actually going to be running our fine-tuned model.
  * *opts.expDir* - set this to the directory where you want your inferences to be saved. This is usually in data/modelzoo/[folderName].
    * as mentioned earlier, it is necessary that imdb.mat is copied to this folder so that the model knows what the test set is
  * *opts.dataDir* - set this to the directory where the full set of jpg images are held. This is generally a folder full of jpgs as a subdirectory of data/satellite
  * *opts.modelPath* - set this equal to the location of the find-tuned neural network model that was produced in the fine-tuning step. This is in the data/fcn8s-satellite_XXXX folder, as the net-epoch-###.mat file specified earlier.
* Running this script will give a set of error metrics

## Running

* Set this up identically to the Testing section above. This is essentially the same thing, but it saves the inferences to images that are not part of the training or test sets

## About

This code was originally developed by

* Sebastien Ehrhardt
* Andrea Vedaldi

The original FCN models can be downloaded from the MatConvNet
[model repository](http://www.vlfeat.org/matconvnet/pretrained/).
The original MatConvNet FCN code can be downloaded from
the [corresponding vlfeat repository](https://github.com/vlfeat/matconvnet-fcn).

Universal Access additions to the code and documentation was performed by

* Stephen Lee

## References

'Fully Convolutional Models for Semantic Segmentation', *Jonathan
Long, Evan Shelhamer and Trevor Darrell*, CVPR, 2015
([paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)).

## Changes
* v0.9.1.UA -- Adding code to train the models particularly for building extraction
* v0.9.1 -- Bugfixes.
* v0.9   -- Initial release. FCN32s and FCN16s work well.
