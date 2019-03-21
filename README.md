# Semi-Convolutions for Keras

This repository includes a keras implementation of semi-convolutional operators from Novotny et al., "Semi-convolutional Operators for Instance Segmentation".
([Paper](https://arxiv.org/abs/1807.10712), [Poster](http://www.robots.ox.ac.uk/~david/posters/poster_semi.png)).

In general, the idea of semi-convolutions consists of mixing the convolutional operator with a non-convolutional information, e.g. the global pixel location, to solve the dilemma of convolutional coloring and translational invariance.
Additionally, the authors got (small) improvements for Mask-RCNN on Microsoft COCO.

The approach is closely related to the so-called Coord-Conv method ([Paper](https://arxiv.org/abs/1807.03247), [Code](https://github.com/titu1994/keras-coordconv)).
In contrast to Coord-Conv, the information of global pixel location is integrated by Addition instead of Concatenation.
The paper aims at instance segmentation instead of simple object detection (proposal-based instance segmentation as well as semantic-based embeddings!).

## Contents
- semiconv.py: Semi-Convolutional Layer, some examples of mixing functions.
- test_semiconv.py: Simple Python Script to visualize the Semi-Convolutional concept.

## Notes
- Tensorflow is used as keras backend.
- Currently, the implementation supports the "channels_last" data format and 2d-convolutions.

## Requirements
- keras (tested with version 2.2.4.)
- tensorflow (tested with version 1.12.0)
- numpy, matplotlib
- tested with Python 3.6.

## Todo
- add kernel for semi-convolutional embedding
- add embedding loss

