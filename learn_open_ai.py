import math
import random
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pyboy import PyBoy, WindowEvent
import sys
from collections import Counter
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

DEBUG = False

# Map
WIDTH = 10
HEIGHT = 10

# Training Constants
ITERS = 20
NUM_EPISODES = 100
EPOCHS = 10
MAX_STEPS = 1000

# Poker Constants
INITIAL_STACK = 200
SB = 1

# PPO Constants
CLIP_EPSILON = 0.2



# PRESS = {0: WindowEvent.PRESS_ARROW_UP, 1: WindowEvent.PRESS_ARROW_RIGHT, 2: WindowEvent.PRESS_ARROW_DOWN, 3: WindowEvent.PRESS_ARROW_LEFT, 4: WindowEvent.PRESS_BUTTON_A, 5: WindowEvent.PRESS_BUTTON_B}
# RELEASE = {0: WindowEvent.RELEASE_ARROW_UP, 1: WindowEvent.RELEASE_ARROW_RIGHT, 2: WindowEvent.RELEASE_ARROW_DOWN, 3: WindowEvent.RELEASE_ARROW_LEFT, 4: WindowEvent.RELEASE_BUTTON_A, 5: WindowEvent.RELEASE_BUTTON_B}
PRESS = {0: WindowEvent.PRESS_ARROW_UP, 1: WindowEvent.PRESS_ARROW_RIGHT, 2: WindowEvent.PRESS_ARROW_DOWN, 3: WindowEvent.PRESS_ARROW_LEFT}
RELEASE = {0: WindowEvent.RELEASE_ARROW_UP, 1: WindowEvent.RELEASE_ARROW_RIGHT, 2: WindowEvent.RELEASE_ARROW_DOWN, 3: WindowEvent.RELEASE_ARROW_LEFT}
WIDTH = 240
HEIGHT = 256
INPUT_DIM = WIDTH*HEIGHT*3
OUPUT_DIM = 7 # up, right, down, left, a, b

# class PolicyNetwork(nn.Module):
#     def __init__(self, INPUT_DIM, OUPUT_DIM):
#         super(PolicyNetwork, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(INPUT_DIM, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, OUPUT_DIM),
#             nn.Softmax(dim=-1)
#         )

#     def forward(self, x):
#         return self.fc(x)

# class ValueNetwork(nn.Module):
#     def __init__(self, INPUT_DIM):
#         super(ValueNetwork, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(INPUT_DIM, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )

#     def forward(self, x):
#         return self.fc(x)


class PolicyNetworkConv(nn.Module):
    def __init__(self, OUPUT_DIM):
        super(PolicyNetworkConv, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(61440, 512),
            nn.ReLU(),
            nn.Linear(512, OUPUT_DIM),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

class ValueNetworkConv(nn.Module):
    def __init__(self):
        super(ValueNetworkConv, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(61440, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.fc(x)




if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS activated")
else:
    device = torch.device("cpu")
    print("MPS device not found.")

def ppo_update(states, actions, returns, old_probs, policy_net, value_net, policy_optimizer, value_optimizer, clip_epsilon=CLIP_EPSILON):
    
    states = torch.stack(states, 0).reshape(len(states), 3, HEIGHT, WIDTH)
    actions = torch.tensor(actions, device=device).reshape(-1, 1)
    old_probs = torch.tensor(old_probs, device=device).reshape(-1, 1)
    returns = torch.tensor(returns, device=device, dtype=torch.float32).reshape(-1, 1)
    value_losses = []
    policy_losses = []


    
    
    for _ in range(EPOCHS):
        # Compute current action probabilities
        current_probs = policy_net(states).gather(1, actions)
        ratio = current_probs / old_probs

        # Compute values
        values = value_net(states)
        advantage = returns - values.detach()

        # Compute surrogate objective
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        # Update policy
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Update value function
        value_loss = nn.MSELoss()(returns, values)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        value_losses.append(value_loss.cpu().detach())
        policy_losses.append(policy_loss.cpu().detach())

    return value_losses, policy_losses, policy_net, value_net

def collect_trajectories(env, policy_net, num_episodes, max_steps, gamma=0.99):
    states, actions, rewards, old_probs, returns = [], [], [], [], []

    with torch.no_grad():
        for episode in range(num_episodes):
            episode_rewards = []
            episode_states = []
            episode_actions = []
            episode_probs = []

            done = True
            new_state, _ = env.reset()
            for step in range(max_steps):
                # X = torch.tensor(np.asarray(env.game_area(), dtype=np.float32)).reshape(1, -1).to(device)
                state = torch.tensor(new_state.copy(), device=device, dtype=torch.float32).reshape(1, 3, HEIGHT, WIDTH) / 252
                action_probs = policy_net(state).cpu().detach().numpy()[0]                
                action_num = np.random.choice(len(action_probs), p=action_probs)
                
                new_state, reward, terminated, truncated, info = env.step(action_num)
                done = terminated or truncated

                episode_states.append(state)
                episode_actions.append(action_num)
                episode_probs.append(action_probs[action_num])
                episode_rewards.append(reward)

                if done:
                    break
            
            G = 0
            episode_returns = np.zeros_like(episode_rewards, dtype=float)
            for t in reversed(range(len(episode_rewards))):
                G = episode_rewards[t] + gamma * G
                episode_returns[t] = G

            
            # Append episode values to global lists
            states += episode_states
            actions += episode_actions
            rewards += episode_rewards
            old_probs += episode_probs
            returns += list(episode_returns)

    return states, actions, rewards, old_probs, returns


def train():
    if "DEBUG" in sys.argv:
        NUM_EPISODES = 2
        ITERS = 3
        MAX_STEPS = 100
    else:
        NUM_EPISODES = 3
        ITERS = 30
        MAX_STEPS = 1000

    policy_net = PolicyNetworkConv(OUPUT_DIM).to("mps")
    value_net = ValueNetworkConv().to("mps")

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)
    value_optimizer = optim.Adam(value_net.parameters(), lr=0.0001)

    env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    value_losses, policy_losses = [], []
    
    try:
        for i in range(ITERS):
            print(f"Round {i+1}/{ITERS}")
            states, actions, rewards, old_probs, returns = collect_trajectories(env, policy_net, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, gamma=0.99)

            value_loss, policy_loss, policy_net, value_net = ppo_update(states, actions, returns, old_probs, policy_net, value_net, policy_optimizer, value_optimizer)

            value_losses += value_loss
            policy_losses += policy_loss

            if len(Counter(np.array(actions).reshape(-1))) < 3:
                print(Counter(np.array(actions).reshape(-1)))
            print(f"Val loss: {value_loss[-1]}, Policy loss: {policy_loss[-1]}")
    except KeyboardInterrupt:
        pass
    



    fig, ax = plt.subplots(ncols=2, figsize=(10,4))

    ax[0].plot(value_losses)
    ax[1].plot(policy_losses)

    ax[0].set_title("Value losses")
    ax[1].set_title("Policy losses")


    plt.show()

    env_test = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
    env_test = JoypadSpace(env_test, SIMPLE_MOVEMENT)
    

    while True:
        done = False
        new_state, _ = env_test.reset()
        while not done:
            state = torch.tensor(new_state.copy(), device=device, dtype=torch.float32).reshape(1, 3, HEIGHT, WIDTH)

            action_probs = policy_net(state).cpu().detach().numpy()[0]                
            # print(action_probs)
            action_num = np.argmax(action_probs)
            
            new_state, reward, terminated, truncated, info = env_test.step(action_num)
            done = terminated or truncated

if __name__ == "__main__":
    train()