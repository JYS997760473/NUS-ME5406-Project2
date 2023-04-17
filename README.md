# NUS-ME5406-Project2

NUS ME5406 Deep Learning for Robotics Project 2

## Install

This project is based on OpenAI gym

First install gym environment:

```bash
pip install gymnasium

pip install box2d pygame
```

## Modification

In ARC computer: for `pkg_resources`: comment `__init__.py` line 121 and line 2338
to avoid the warning.

## OpenAI Gym Spinning Up

python version: python3.6

Operation System: Ubuntu 18 and Ubuntu 20

## Install OpenMPI

```bash
sudo apt-get update && sudo apt-get install libopenmpi-dev
```

## Install Spinning Up

```bash
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .
```

## Install Bugs Fixed

First cannot install box2d-py, then run `brew install swig`, install swig first, then run
`sudo apt-get install swig build-essential python-dev python3-dev`, then `pip install gym[box2d]` works.

## Result

Final trained agent is shown in this gif:

![Alt Text](./GIF/episode1.gif)