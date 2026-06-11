import gymnasium as gym
from gymnasium.utils.play import play, PlayPlot

# Create the environment
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# Callback for the plotter
def callback(obs_t, obs_t1, action, rew, terminated, truncated, info):
    return [rew]

# Plot the reward live
plotter = PlayPlot(callback,150,["reward"])
# Run interactive play
play(
    env,
    keys_to_action={
        "w": 2,  # Main engine
        "a": 3,  # Left engine
        "d": 1   # Right engine
    },
    noop=0,
    callback=plotter.callback
)
