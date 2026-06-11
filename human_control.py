import gymnasium as gym
import numpy as np
import time

env = gym.make("Humanoid-v4", render_mode="human")

obs, info = env.reset()

print("Observation Space:")
print(env.observation_space)

print("\nAction Space:")
print(env.action_space)

while True:

    # Alle Gelenke auf 0
    action = np.zeros(env.action_space.shape[0])

    # Beispiel:
    # Erstes Gelenk leicht bewegen
    action[0] = 0.2

    obs, reward, terminated, truncated, info = env.step(action)
    print(obs)
    print(len(obs))

    env.render()

    time.sleep(0.01)

    if terminated or truncated:
        obs, info = env.reset()