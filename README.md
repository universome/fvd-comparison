# About the repo

In this repo, we demonstrate that the FVD implementation from StyleGAN-V paper is equivalent to the [original one](https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py) when the videos are already loaded into memory and resized to a necessary resolution.
The main difference of our FVD evaluation protocol from the paper is that we strictly specify how data should be processed, clips sampled, etc.

# Why did we implement FVD ourselves?

The problem with the [original implementation](https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py) is that it does not handle:
- data processing: in which format videos are being stored (JPG/PNG directories of frames or MP4, etc.), how frames are being resized, normalized, etc.
- clip sampling strategy: how clips are being selected (from the beginning of the video, or randomly, with which framerate, how many clips per video, etc.)
- how many fake and real videos should be used

That's why every project computes FVD in their own way and this leads to a lot of discrepancies.
In StyleGAN-V, we seek to establish a unified evaluation pipeline.

Also, the original tensorflow snippet is implemented in TensorFlow v1, which final release was done on January 6, 2021 (i.e. more than a year ago) and it won't be updated since then: https://github.com/tensorflow/tensorflow/releases/tag/v1.15.5

# How to launch the comparison

We provide two comparisons:
1. Comparison between `tf.hub`'s I3D model and our `torchscript` port to demonstrate that our port is a *perfectly precise* copy (up to numerical precision) of `tf.hub`'s one.
2. Comparison between FVD metrics itself. It is done by generating two dummy datasets of 256 videos each with two different random seeds.

### Installing dependencies
We put all the dependencies used into `requirements.txt`.
You can install them by running:
```
pip install -r requirements.txt
```

### 1. Launching models' comparison
To compare the models between each other (in terms of L2 distance of their output), run:
```
python compare_models.py
```
In our case, it gives the output:
```
L_2 difference is 0.00026316816207043225
```
Which means that both models perform equivalent operations (note that even two equivalent convolutional layers in TF and PyTorch would produce slightly different outputs due to numerical percision).

### 2. Launching metrics' comparison
To compare the metrics between each other, run:
```
python compare_models.py
```

On our machine, this gives the output:
```
[FVD scores] Theirs: 10.13808536529541. Ours: 10.138084766713924
```
So, the difference is 1e-6, which is negligible.

Note: computing FVD for TensorFlow's implementation might take time since they use exact the square root.
In our case, we use a *very* accurate [approximate square root](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html).
