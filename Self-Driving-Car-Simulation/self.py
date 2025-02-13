from collections import deque

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython import display
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Initialize the environment
env = gym.make("CarRacing-v2", render_mode="rgb_array")
obs, _ = env.reset()

# Parameters
replay_buffer_size = 10000  # Size of the replay buffer
num_episodes = 10  # Number of episodes to run for data collection
max_steps_per_episode = 500  # Max steps per episode
batch_size = 32  # Batch size for training
gamma = 0.99  # Discount factor
learning_rate = 0.001  # Learning rate

# Initialize the replay buffer
replay_buffer = deque(maxlen=replay_buffer_size)


def preprocess_frame(frame):
    """Preprocess the game frame."""
    if frame is None or not isinstance(frame, np.ndarray):
        return np.zeros((84, 84))  # Return a blank frame if invalid

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = gray / 255.0
    processed_frame = cv2.resize(gray, (84, 84))
    return processed_frame


def normalize_rewards(rewards):
    """Normalize the rewards."""
    rewards = np.array(rewards)
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-5)
    return rewards


def build_model(input_shape, action_space):
    """Build a simple convolutional neural network model."""
    model = Sequential(
        [
            Conv2D(32, (8, 8), strides=4, activation="relu", input_shape=input_shape),
            Conv2D(64, (4, 4), strides=2, activation="relu"),
            Conv2D(64, (3, 3), strides=1, activation="relu"),
            Flatten(),
            Dense(512, activation="relu"),
            Dense(action_space.shape[0], activation="linear"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def train_dqn_model(model, replay_buffer, batch_size, gamma):
    """Train the DQN model."""
    if len(replay_buffer) < batch_size:
        return

    batch = np.random.choice(len(replay_buffer), batch_size, replace=False)
    for idx in batch:
        state, action, reward, next_state, done = replay_buffer[idx]

        target = reward
        if not done:
            target += gamma * np.amax(
                model.predict(np.expand_dims(next_state, axis=0), verbose=0)
            )

        target_f = model.predict(np.expand_dims(state, axis=0), verbose=0)
        target_f[0][np.argmax(action)] = target

        model.fit(np.expand_dims(state, axis=0), target_f, epochs=1, verbose=0)


# Data collection
for episode in range(num_episodes):
    obs, _ = env.reset()
    state = preprocess_frame(obs)
    done = False
    step = 0

    while not done and step < max_steps_per_episode:
        action = env.action_space.sample()  # Random action
        action[2] = 1.0  # Set maximum acceleration
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = preprocess_frame(next_obs)

        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        step += 1

        if step % 50 == 0:  # Render every 50 steps to reduce frequency
            plt.imshow(next_obs)
            display.display(plt.gcf())
            display.clear_output(wait=True)

    print(f"Episode {episode + 1}/{num_episodes} finished after {step + 1} steps")

# Convert replay buffer to NumPy arrays for easier handling
states, actions, rewards, next_states, dones = zip(*replay_buffer)
states = np.array(states)
actions = np.array(actions)
rewards = np.array(rewards)
next_states = np.array(next_states)
dones = np.array(dones)

# Normalize rewards
rewards = normalize_rewards(rewards)

# Shuffle the data
indices = np.arange(len(replay_buffer))
np.random.shuffle(indices)

states = states[indices]
actions = actions[indices]
rewards = rewards[indices]
next_states = next_states[indices]
dones = dones[indices]

print(f"Collected {len(replay_buffer)} experiences")

# Save the collected data as numpy files for future use
np.save("states.npy", states)
np.save("actions.npy", actions)
np.save("rewards.npy", rewards)
np.save("next_states.npy", next_states)
np.save("dones.npy", dones)

# Build the DQN model
input_shape = (84, 84, 1)
model = build_model(input_shape, env.action_space)

# Train the DQN model with the collected data
train_dqn_model(model, replay_buffer, batch_size, gamma)

# Save the trained model using the new Keras format
model.save("dqn_model.keras")  # Updated file format
