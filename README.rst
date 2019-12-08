.. |copy| unicode:: 0xA9
.. |---| unicode:: U+02014

=====================
MuZero implementation
=====================

This repository is a Python implementation of the MuZero algorithm.
It is based upon the `pre-print paper`__ and the `pseudocode`__ describing the Muzero framework.
You can easily train your own MuZero, more specifically for one player and non-image based environments (such as `CartPole`__).
If you wish to train Muzero on other kinds of environments, this codebase can be used with slight modifications.

__ https://arxiv.org/abs/1911.08265
__ https://arxiv.org/src/1911.08265v1/anc/pseudocode.py
__ https://gym.openai.com/envs/CartPole-v1/


**DISCLAIMER**: this code is early research code. What this means is:

- Silent bugs may exist.
- It may not work reliably on other environments or with other hyper-parameters.
- The code quality and documentation are quite lacking, and much of the code might still feel "in-progress".
- The training and testing pipeline is not very advanced, which is not the goal of this repository.

Dependencies
============

We run this code using:

- Conda **4.7.12**
- Python **3.7**
- Tensorflow **2.0.0**
- Numpy **1.17.3**

Training your MuZero
====================

Currently, this code must be run from the main function in ``muzero.py`` (don't forget to first configure your conda environment).

Training a Cartpole-v1 bot
--------------------------

To train a model, please follow these steps:

1) Create or modify an existing configuration of Muzero in ``config.py``.

2) Be sure to call the right configuration at the first line of the main inside ``muzero.py``.

3) Run the main function: ``python muzero.py``.
