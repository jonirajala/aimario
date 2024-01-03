import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import Counter
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from torch.utils.data import TensorDataset, DataLoader
import cv2
from model import Model
from torch.distributions import Categorical
import torch.nn.functional as F

DEBUG = False

# Training Constants
ITERS = 20
NUM_EPISODES = 100
EPOCHS = 10
MAX_STEPS = 1000
BATCH_SIZE = 32
LR = 0.0001

# PPO Constants
CLIP_EPSILON = 0.2
BETA = 0.3
GAMMA = 0.95
TAU = 1.0

# Env constants
WIDTH = 84
HEIGHT = 84
INPUT_DIM = WIDTH*HEIGHT*3
OUPUT_DIM = 7 

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS activated")
else:
    device = torch.device("cpu")
    print("MPS device not found.")

def ppo_update(states, actions, returns, old_probs, model, optimizer, advantages):
    states = torch.stack(states, 0).reshape(len(states), 3, HEIGHT, WIDTH)
    actions = torch.tensor(actions, device=device).reshape(-1, 1)
    old_probs = torch.tensor(old_probs, device=device).reshape(-1, 1)
    advantages = torch.tensor(advantages, device=device).reshape(-1, 1)
    returns = torch.tensor(returns, device=device, dtype=torch.float32).reshape(-1, 1)
    
    dataset = TensorDataset(states, actions, old_probs, returns, advantages)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    losses = []

    for _ in range(EPOCHS):
        for state_batch, action_batch, old_prob_batch, return_batch, advantage_batch in data_loader:
            # Compute current action probabilities
            logits, values = model(state_batch)
            new_policy = F.softmax(logits, dim=1)
            new_m = Categorical(new_policy)
            new_log_policy = new_m.log_prob(action_batch)
            ratio = torch.exp(new_log_policy - old_prob_batch)

            # Compute surrogate objective
            surr1 = ratio * advantage_batch
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantage_batch
            policy_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.smooth_l1_loss(return_batch.squeeze(), values.squeeze())

            entropy_loss = torch.mean(new_m.entropy())

            tot_loss = policy_loss + critic_loss - entropy_loss * BETA

            # Update policy
            optimizer.zero_grad()
            tot_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            losses.append(tot_loss.cpu().detach())

    return losses, model

def collect_trajectories(env, model, num_episodes, max_steps):
    states, actions, rewards, old_probs, returns, values, advantages = [], [], [], [], [], [], []

    with torch.no_grad():
        for episode in range(num_episodes):
            episode_rewards = []
            episode_states = []
            episode_actions = []
            episode_probs = []
            episode_values = []
            episode_dones = []

            done = False
            new_state, _ = env.reset()
            last_score = 0
            for step in range(max_steps):
                new_state = cv2.resize(new_state, (84, 84))
                state = torch.tensor(new_state.copy(), device=device, dtype=torch.float32).reshape(1, 3, HEIGHT, WIDTH) / 255
                logits, value = model(state)
                policy = F.softmax(logits, dim=1)
                old_m = Categorical(policy)
                action = old_m.sample()
                old_log_policy = old_m.log_prob(action)
                new_state, reward, terminated, truncated, info = env.step(action.item())
                reward += (info["score"] - last_score) / 40.
                last_score = info["score"]
                
                done = terminated or truncated
                if done:
                    if info["flag_get"]:
                        reward += 50
                    else:
                        reward -= 50
                
                reward /= 10.

                episode_values.append(value)
                episode_states.append(state)
                episode_actions.append(action.cpu())
                episode_probs.append(old_log_policy)
                episode_rewards.append(reward)
                episode_dones.append(done)

                if done:
                    break
                
            new_state = cv2.resize(new_state, (84, 84))
            state = torch.tensor(new_state.copy(), device=device, dtype=torch.float32).reshape(1, 3, HEIGHT, WIDTH) / 255
            _, next_value, = model(state)
            next_value = next_value.squeeze()
            gae = 0
            R = []
            for value, reward, done in list(zip(episode_values, episode_rewards, episode_dones))[::-1]:
                gae = gae * GAMMA * TAU
                gae = gae + reward + GAMMA * next_value.detach() * (1 - done) - value.detach()
                next_value = value
                R.append(gae + value)
            R = R[::-1]
            episode_advantages = [R[i] - episode_values[i] for i in range(len(R))]

            states += episode_states
            actions += episode_actions
            rewards += episode_rewards
            old_probs += episode_probs
            values += episode_values
            returns += R
            advantages += episode_advantages

    return states, actions, rewards, old_probs, returns, advantages


def train():
    if "DEBUG" in sys.argv:
        NUM_EPISODES = 2
        ITERS = 3
        MAX_STEPS = 100
    else:
        NUM_EPISODES = 3
        ITERS = 50
        MAX_STEPS = 512

    model = Model(OUPUT_DIM).to("mps")
    optimizer = optim.Adam(model.parameters(), lr=LR)
    env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    tot_losses = []
    
    try:
        for i in range(ITERS):
            states, actions, rewards, old_probs, returns, advantages = collect_trajectories(env, model, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)
            losses, model = ppo_update(states, actions, returns, old_probs, model, optimizer, advantages) 
            tot_losses += losses

            print(f"Round {i+1}/{ITERS} -- loss: {sum(losses) / len(losses):.2f}")
            if len(Counter(np.array(actions).reshape(-1))) < 3:
                print(Counter(np.array(actions).reshape(-1)))

    except KeyboardInterrupt:
        pass

    fig, ax = plt.subplots(ncols=1, figsize=(10,4))
    ax.plot(tot_losses)
    ax.set_title("losses")
    plt.show()

    env_test = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
    env_test = JoypadSpace(env_test, SIMPLE_MOVEMENT)

    while True:
        done = False
        state, _ = env_test.reset()
        while not done:
            state = cv2.resize(state, (84, 84))
            state = torch.tensor(state.copy(), device=device, dtype=torch.float32).reshape(1, 3, HEIGHT, WIDTH) / 255
            logits, _ = model(state)
            policy = F.softmax(logits, dim=1)
            action = torch.argmax(policy).item()
            state, reward, terminated, truncated, info = env_test.step(action)
            done = terminated or truncated

if __name__ == "__main__":
    train()