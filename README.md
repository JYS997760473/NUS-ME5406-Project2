# NUS-ME5406-Project2

NUS ME5406 Deep Learning for Robotics Project 2

This project is based on OpenAI Gym

## Install

This project is built on `Python3.8`

It has been tested on `MacOS 13.0`, `Ubuntu 18.04` and `Ubuntu 20.04`

Note: if you run it on `MacOS 13.0`, please do not render the environment, because MacOS 13.0 now does not support `OpenGL`

First install the requirements of this project by `pip`

```bash
pip install -r requirements.txt
```

### Potential Bugs Fix during installing

If your PC cannot install `box2d-py`, please try to run `brew install swig`, install swig first, then run
`sudo apt-get install swig build-essential python-dev python3-dev`, then `pip install gym[box2d]` normally will work.

## Run this project

### Train the agent

Run this to train the agent:

```bash
python train.py --exp_name <experiment name> --gamma <gamma in SAC> --render <whether render the environment> --batch_size <batch size in FIFO buffer replay experience>
```

For example, run:

```bash
python train.py --exp_name my_experiment --gamma 0.99 --episode 2000 --render True --batch_size 256
```

After training, an `json` file `output.json`, `model.pt`, and `vars.pkl` will be generated under `./model/my_experiment/` directory.

#### Plot training results

If you want to plot the training results, run:

```bash
python plot.py --exp_name <experiment name>
```

For example, run

```bash
python plot.py --exp_name my_experiment
```

### Test the pre-trained model

There are already several pre-trained models under the `./model` directory. Run this to test the pre-trained model:

```bash
python test.py --exp_name <model name chosen> --episode <number of episodes want to test>
```

For example, run:

```bash
python test.py --exp_name my_experiment --episode 100 
```

## Demo

Final trained agent is shown in this gif:

![Alt Text](./GIF/episode1.gif)

## Training Experiment Results Example

### batch size = 100

Reward:
![alt-text-1](./model/arc4.17.22:292000/reward.png)

Number of total successful episode:
![alt-text-2](./model/arc4.17.22:292000/success.png)

Distribution of successful episode during training
![alt-text-3](./model/arc4.17.22:292000/successdistribute.png)

Time consumption of each episode during training
![alt-text-4](./model/arc4.17.22:292000/timebar.png)

### batch size = 256

Reward:
![alt-text-5](./model/)

Number of total successful episode:
![alt-text-6](./model/arc4.17.22:292000/success.png)

Distribution of successful episode during training
![alt-text-7](./model/arc4.17.22:292000/successdistribute.png)

Time consumption of each episode during training
![alt-text-8](./model/arc4.17.22:292000/timebar.png)

## Testing Experiment Results Example
