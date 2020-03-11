
# MuZero

This repository is a Python implementation of the MuZero algorithm.
It is based upon the [pre-print paper](https://arxiv.org/abs/1911.08265) and the 
[pseudocode](https://arxiv.org/src/1911.08265v1/anc/pseudocode.py) describing the 
Muzero framework. Neural computations are implemented with Tensorflow.

You can easily train your own MuZero, more specifically for one player and non-image based environments (such as 
[CartPole](https://gym.openai.com/envs/CartPole-v1/)).
If you wish to train Muzero on other kinds of environments, this codebase can be used with slight modifications.

**DISCLAIMER**: this code is early research code. What this means is:

- Silent bugs may exist.
- It may not work reliably on other environments or with other hyper-parameters.
- The code quality and documentation are quite lacking, and much of the code might still feel "in-progress".
- The training and testing pipeline is not very advanced.

### Dependencies

We run this code using:

- Conda **4.7.12**
- Python **3.7**
- Tensorflow **2.0.0**
- Numpy **1.17.3**

Do the following on a linux gcloud instance to install python 3.7

source: https://stackoverflow.com/questions/53468831/how-do-i-install-python-3-7-in-google-cloud-shell

install pyenv to install python on persistent home directory:

`curl https://pyenv.run | bash`

add to path:

`echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc`

`echo 'eval "$(pyenv init -)"' >> ~/.bashrc`

`echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc`

Enter a new bash session to use pyenv

Install python 3.7.4 and make default:

`pyenv install 3.7.4`

`pyenv global 3.7.4`

verify the installation:

`python --version`

Then, due to some [graphviz bug](https://github.com/pydot/pydot/issues/91#issuecomment-229639358), 
you need to manually install it:

`sudo apt-get install graphviz`

### Training your MuZero

This code must be run from the main function in ``muzero.py`` (don't forget to first configure your conda environment).

Training a Cartpole-v1 bot

To train a model, please follow these steps:

1) Create or modify an existing configuration of Muzero in ``config.py``.

2) Call the right configuration inside the main of ``muzero.py``.

3) Run the main function: ``python muzero.py``.

Training on an other environment

To train on a different environment than Cartpole-v1, please follow these additional steps:

1) Create a class that extends ``AbstractGame``, this class should implement the behavior of your environment.
For instance, the ``CartPole`` class extends ``AbstractGame`` and works as a wrapper upon 
[`gym CartPole-v1`](https://gym.openai.com/envs/CartPole-v1/).
You can use the ``CartPole`` class as a template for any gym environment.

2) **This step is optional** (only if you want to use a different kind of network architecture or value/reward transform).
Create a class that extends ``BaseNetwork``, this class should implement the different networks (representation, value, policy, reward and dynamic) and value/reward transforms.
For instance, the ``CartPoleNetwork`` class extends ``BaseNetwork`` and implements fully connected networks.

3) **This step is optional** (only if you use a different value/reward transform).
You should implement the corresponding inverse value/reward transform by modifying the ``loss_value`` and ``loss_reward`` function inside ``training.py``.

Differences from the paper

This implementation differ from the original paper in the following manners:

- We use fully connected layers instead of convolutional ones. This is due to the nature of our environment (Cartpole-v1) which as no spatial correlation in the observation vector.
- We don't scale the hidden state between 0 and 1 using min-max normalization. Instead we use a tanh function that maps any values in a range between -1 and 1.
- We do use a slightly simple invertible transform for the value prediction by removing the linear term.
- During training, samples are drawn from a uniform distribution instead of using prioritized replay.
- We also scale the loss of each head by 1/K (with K the number of unrolled steps). But, instead we consider that K is always constant (even if it is not always true).
