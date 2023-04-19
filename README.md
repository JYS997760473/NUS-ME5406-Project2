# NUS-ME5406-Project2

NUS ME5406 Deep Learning for Robotics Project 2

## Install

This project is based on OpenAI Gym

This project is built on Python3.8.

It has been tested on MacOS 13.0, Ubuntu 18.04 and Ubuntu 20.04.

Note: if you run it on MacOS 13.0, please do not render the environment, because MacOS 13.0 now does not support `OpenGL`

First install the requirements of this project by `pip`

```bash
pip install -r requirements.txt
```

### Install Bugs Fixed

If your PC cannot install box2d-py, please try to run `brew install swig`, install swig first, then run
`sudo apt-get install swig build-essential python-dev python3-dev`, then `pip install gym[box2d]` normally will work.

## Result

Final trained agent is shown in this gif:

![Alt Text](./GIF/episode1.gif)
