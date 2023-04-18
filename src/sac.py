import numpy as np
import torch
from torch.optim import Adam
from copy import deepcopy
from src import mlp
import time
import itertools
from src.replaybuffer import ReplayBuffer
import joblib
import os.path as osp, os
import json
    

class SAC:
    def __init__(self, env, exp_name, actor_critic=mlp.MLPActorCritic, seed=0, replay_size=int(1e6),
                 gamma=0.99, polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, num_episode=500) -> None:
        
        print(f"initializing SAC...")

        torch.set_num_threads(torch.get_num_threads())
        print(f"using {torch.get_num_threads()} threads for parallelizing CPU operations")
        self.env = env
        self.actor_critic = actor_critic
        self.seed = seed
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.lr = lr
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_episode = num_episode
        self.output_dir = "./model/"+exp_name+"/"
        try:
            os.mkdir(self.output_dir)
        except FileExistsError:
            pass
        self.success = 0
        
        # set random seed
        torch.manual_seed(seed=self.seed)
        np.random.seed(seed=self.seed)

        # set shapes of actions and states of the environment
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        # Create actor-critic module and target networks
        self.ac = self.actor_critic(env.observation_space, env.action_space)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)

    # Function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info
    
    # update data
    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)
    
    def run(self):
        print(f"running experiment...")
        # loop
        json_list = []
        total_steps = 0
        total_start_time = time.time()
        for episode in range(self.num_episode):
            # one episode
            obs, info = self.env.reset()
            self.env.render()
            episode_reward = 0
            episode_steps = 0
            done = False
            start_time = time.time()
            episode_success = False
            # one episode
            while not done:
                if episode < 5:
                    action = self.env.action_space.sample()
                else:
                    action = self.get_action(obs)
                episode_steps += 1
                # take action to next state
                obs2, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated or episode_steps > 5000
                # refine the reward structure
                if reward == -100:
                    reward = -1
                reward = reward * 10
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                # Store experience to replay buffer
                self.replay_buffer.store(obs=obs, act=action, rew=reward, next_obs=obs2, done=done)
                # replay_buffer.store(obs, action, reward, obs2, done)
                obs = obs2
                # update
                if total_steps % 50 == 0 and total_steps > 1000:
                    for i in range(50):
                        batch = self.replay_buffer.sample_batch(self.batch_size)
                        self.update(data=batch)

            if terminated == True and truncated == False and reward > 0:
                # reach the end, this episode is successful
                episode_success = True
                self.success += 1
            # save vars.pkl environment
            joblib.dump({'env': self.env}, osp.join(self.output_dir, 'vars.pkl'))

            # save model.pt
            fname = osp.join(self.output_dir, 'model.pt')
            torch.save(self.ac, fname)

            print(f"total steps: {total_steps}, episode:{episode}, steps:{episode_steps}, reward:{episode_reward},\
                time: {time.time()-start_time}, episode success: {episode_success}, success:{self.success}, \
                  truncated:{truncated}, terminated:{terminated}\
                  final step reward:{reward}, total time used:{time.time()-total_start_time}")
            
            # record log to json file
            one_episode_dict = {'episode':episode, 'steps':episode_steps, 'reward':episode_reward, 'time':\
                                time.time()-start_time, 'success':episode_success, 'success_num':self.success, \
                                    'truncated':truncated, 'terminated':\
                                terminated, 'final_step_reward':reward, 'total_steps':total_steps, 'total_time':\
                                    time.time()-total_start_time}
            json_list.append(one_episode_dict)

        # end all episodes, record output
        json_dict = {'episodes': json_list}
        json_file = osp.join(self.output_dir, 'output.json')
        file = open(json_file, 'w')
        json.dump(json_dict, file, indent=4)
        file.close()
            
        # close environment
        print(f"close environment")
        self.env.close()