# S-TPNet
PyTorch demo code for "Spatial-Temporal Pyramid Based Convolutional Neural Network for Action Recognition" 

We propose a new spatial pyramid module to take full use of inherent multi-scale information of CNNs with nearly cost-free by which a bottom-up architecture with lateral connections is constructed for combining high-, mid-, low-level representations into a frame-level feature elaborately. To capture more comprehensive long-range temporal structure, we also propose a new temporal pyramid module in which frame-level features are reused by various pooling approaches to get different time-grained features of snippets efficiently. Followed by snippet-relation reasoning, different snippet-relations are derived and accumulated for the final prediction. Unifying both spatial and temporal pyramid modules, a novel network, Spatial-Temporal Pyramid Network(S-TPNet), is proposed to extract spatial-temporal pyramid features for action recognition in videos. Unlike previous models which boost performance at the cost of computation, S-TPNet can be trained in an end-to-end fashion with great efficiency on one GPU. S-TPNet displays significant performance improvements compared with existing frameworks and obtains competitive performance with the state-of-the-art.

# Dependencies

Ubuntu 16.04

Python 3.6

PyTorch 0.3.1

NumPy

Pandas

PIL

tqdm

# Preparation

UCF-101

Download videos and convert from avi to jpg files using Datasets/ucf_video_to_images.py

HMDB-51

Download videos and convert from avi to jpg files using Datasets/hmdb_video_to_images.py



# Coming Soon

The training code

Experimental Details

Results

Visualization
