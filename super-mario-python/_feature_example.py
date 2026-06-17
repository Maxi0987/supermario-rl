"""Produce a concrete state -> features -> Q-values -> action example with a screenshot.

Drives Mario (RIGHT_RUN) up to just before the first pit (tiles 48-49, px 1536-1600),
captures the rendered frame, prints the full 58-dim observation grouped by block, then
runs the trained Q-network on that exact state and reports the chosen action.
"""
import numpy as np
import pygame
import tensorflow as tf

from rl_env import SuperMarioPythonEnv, ACTION_MAP
from rl_game import TILE_X_OFFSETS, TILE_Y_OFFSETS
from dqn_train import build_q_network

MODEL = "training_runs/dqn_20260617_215221/models/q_network_final.weights.h5"
TARGET_X = 1440.0  # tile ~45, one tile before the first pit edge at x=1536

env = SuperMarioPythonEnv(level_name="Level1-1", render_mode="rgb_array", frame_skip=1, max_steps=2000)
obs, info = env.reset()

# Phase 1: bunny-hop right (RIGHT_RUN_JUMP) to clear the pipes between start and the pit.
for _ in range(3000):
    obs, r, term, trunc, info = env.step(7)
    if info["x_pos"] >= 1420 or term or trunc:
        break

# Phase 2: stop jumping so Mario lands; capture the first on-ground frame before the pit.
for _ in range(80):
    obs, r, term, trunc, info = env.step(0)  # NOOP -> land
    if obs[4] == 1.0 or term or trunc:  # obs[4] = onGround
        break

# Screenshot of the exact state.
pygame.image.save(env.game.screen, "feature_example_state.png")

actions = env.get_action_meanings()

print("=== STATE ===")
print(f"x_pos={info['x_pos']:.0f} (tile {info['x_pos']/32:.1f}), y_pos={info['y_pos']:.0f}, "
      f"onGround={obs[4]}, vel.x norm={obs[2]:.3f}")

print("\n=== 58-DIM OBSERVATION ===")
print("Block 1 - Mario state [0:8]:")
labels = ["x (0..1)", "y (0..1)", "vel_x/5", "vel_y/15", "onGround", "inJump", "powerUp", "dist_to_goal"]
for l, v in zip(labels, obs[0:8]):
    print(f"   {l:>14}: {v:+.3f}")

print("Block 2 - last action one-hot [8:16]:", np.round(obs[8:16], 2).tolist())

print(f"Block 3 - solid-tile grid [16:48]  (rows y={list(TILE_Y_OFFSETS)}, cols x={list(TILE_X_OFFSETS)}):")
grid = obs[16:48].reshape(len(TILE_Y_OFFSETS), len(TILE_X_OFFSETS))
header = "        " + " ".join(f"x{o:+d}" for o in TILE_X_OFFSETS)
print(header)
for yo, row in zip(TILE_Y_OFFSETS, grid):
    cells = " ".join(" # " if v > 0.5 else " . " for v in row)
    print(f"   y{yo:+d}:  {cells}")

print("Block 4 - nearest mobs [48:58]:", np.round(obs[48:58], 2).tolist(), "(no enemies -> all 0)")

# Trained network forward pass.
q = build_q_network((obs.shape[0],), len(actions))
q.load_weights(MODEL)
qv = q(np.expand_dims(obs.astype(np.float32), 0), training=False).numpy()[0]

print("\n=== Q-VALUES (trained network) ===")
order = np.argsort(qv)[::-1]
for i in order:
    star = "  <== CHOSEN (argmax)" if i == order[0] else ""
    print(f"   a{i} {actions[i]:>16}: Q={qv[i]:+.4f}{star}")

print(f"\nChosen action: a{order[0]} = {actions[order[0]]}")
env.close()
