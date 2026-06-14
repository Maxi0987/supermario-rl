import argparse
import csv
import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from rl_env import SuperMarioPythonEnv


@dataclass
class DQNConfig:
    episodes: int = 500
    max_steps: int = 1500
    frame_skip: int = 4
    image_size: int = 84
    stack_size: int = 4
    replay_capacity: int = 50000
    warmup_steps: int = 2000
    batch_size: int = 32
    gamma: float = 0.99
    learning_rate: float = 0.00025
    train_every: int = 4
    target_update_every: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100000
    save_every: int = 10
    level_name: str = "Level1-1"
    output_dir: str = "training_runs"
    run_name: str = ""
    render_mode: str = "human"
    fps: int = 0
    seed: int = 42


class ReplayBuffer:
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.state_shape = state_shape
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.index = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size


def preprocess_observation(obs, image_size):
    frame = tf.convert_to_tensor(obs, dtype=tf.uint8)
    frame = tf.image.rgb_to_grayscale(frame)
    frame = tf.image.resize(frame, (image_size, image_size), method="area")
    frame = tf.cast(tf.squeeze(frame, axis=-1), tf.uint8)
    return frame.numpy()


def make_initial_state(frame, stack_size):
    frames = deque(maxlen=stack_size)
    for _ in range(stack_size):
        frames.append(frame)
    return frames, np.stack(frames, axis=-1)


def append_frame(frames, frame):
    frames.append(frame)
    return np.stack(frames, axis=-1)


def build_q_network(input_shape, num_actions):
    num_actions = int(num_actions)
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_actions)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mario_dqn")


def epsilon_by_step(config, global_step):
    fraction = min(1.0, global_step / config.epsilon_decay_steps)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


@tf.function
def train_step(q_network, target_network, optimizer, states, actions, rewards, next_states, dones, gamma):
    states = tf.cast(states, tf.float32)
    next_states = tf.cast(next_states, tf.float32)
    actions = tf.cast(actions, tf.int32)
    rewards = tf.cast(rewards, tf.float32)
    dones = tf.cast(dones, tf.float32)

    next_q_values = target_network(next_states, training=False)
    max_next_q = tf.reduce_max(next_q_values, axis=1)
    targets = rewards + gamma * (1.0 - dones) * max_next_q

    with tf.GradientTape() as tape:
        q_values = q_network(states, training=True)
        action_masks = tf.one_hot(actions, q_values.shape[1])
        selected_q_values = tf.reduce_sum(q_values * action_masks, axis=1)
        loss = tf.keras.losses.Huber()(targets, selected_q_values)

    grads = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
    return loss


def ensure_csv(csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()


CSV_FIELDS = [
    "timestamp",
    "episode",
    "global_step",
    "episode_steps",
    "episode_reward",
    "epsilon",
    "avg_loss",
    "replay_size",
    "x_pos",
    "score",
    "coins",
    "x_reward",
    "score_reward",
    "coin_reward",
    "death_penalty",
    "finish_bonus",
    "dead",
    "finished",
    "model_path",
]


def write_episode_row(csv_path, row):
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def create_run_dir(config):
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.run_name:
        base_name = config.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"dqn_{timestamp}"

    run_dir = output_dir / base_name
    suffix = 2
    while run_dir.exists():
        run_dir = output_dir / f"{base_name}_{suffix}"
        suffix += 1

    (run_dir / "models").mkdir(parents=True)
    return run_dir


def save_run_metadata(run_dir, config, env):
    config_path = run_dir / "config.json"
    config_data = asdict(config)
    config_data["run_dir"] = str(run_dir)
    config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

    actions_path = run_dir / "actions.json"
    actions_path.write_text(
        json.dumps(env.get_action_meanings(), indent=2),
        encoding="utf-8",
    )


def save_model_snapshot(q_network, run_dir, episode):
    weights_path = run_dir / "models" / f"q_network_episode_{episode:05d}.weights.h5"
    q_network.save_weights(weights_path)
    return weights_path


def train(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)

    run_dir = create_run_dir(config)
    csv_path = run_dir / "training_log.csv"
    ensure_csv(csv_path)

    env = SuperMarioPythonEnv(
        level_name=config.level_name,
        render_mode=config.render_mode,
        frame_skip=config.frame_skip,
        max_steps=config.max_steps,
        fps=config.fps,
    )
    save_run_metadata(run_dir, config, env)

    input_shape = (config.image_size, config.image_size, config.stack_size)
    q_network = build_q_network(input_shape, env.action_space.n)
    target_network = build_q_network(input_shape, env.action_space.n)
    target_network.set_weights(q_network.get_weights())
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    replay_buffer = ReplayBuffer(config.replay_capacity, input_shape)

    global_step = 0
    print(f"Run folder: {run_dir}")

    try:
        for episode in range(1, config.episodes + 1):
            obs, info = env.reset()
            first_frame = preprocess_observation(obs, config.image_size)
            frame_stack, state = make_initial_state(first_frame, config.stack_size)

            episode_reward = 0.0
            losses = []
            model_path = ""

            for episode_step in range(1, config.max_steps + 1):
                epsilon = epsilon_by_step(config, global_step)

                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    q_values = q_network(np.expand_dims(state, axis=0), training=False)
                    action = int(tf.argmax(q_values[0]).numpy())

                next_obs, reward, terminated, truncated, info = env.step(action)
                next_frame = preprocess_observation(next_obs, config.image_size)
                next_state = append_frame(frame_stack, next_frame)
                done = terminated or truncated

                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                global_step += 1

                if len(replay_buffer) >= config.warmup_steps and global_step % config.train_every == 0:
                    batch = replay_buffer.sample(config.batch_size)
                    loss = train_step(
                        q_network,
                        target_network,
                        optimizer,
                        *batch,
                        tf.constant(config.gamma, dtype=tf.float32),
                    )
                    losses.append(float(loss.numpy()))

                if global_step > 0 and global_step % config.target_update_every == 0:
                    target_network.set_weights(q_network.get_weights())

                if done:
                    break

            avg_loss = float(np.mean(losses)) if losses else 0.0
            reward_parts = info.get("reward_parts", {})

            if episode % config.save_every == 0 or episode == config.episodes:
                model_path = save_model_snapshot(q_network, run_dir, episode)

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "episode": episode,
                "global_step": global_step,
                "episode_steps": episode_step,
                "episode_reward": round(float(episode_reward), 4),
                "epsilon": round(float(epsilon_by_step(config, global_step)), 6),
                "avg_loss": round(avg_loss, 6),
                "replay_size": len(replay_buffer),
                "x_pos": round(float(info.get("x_pos", 0.0)), 2),
                "score": int(info.get("score", 0)),
                "coins": int(info.get("coins", 0)),
                "x_reward": round(float(reward_parts.get("x_reward", 0.0)), 4),
                "score_reward": round(float(reward_parts.get("score_reward", 0.0)), 4),
                "coin_reward": round(float(reward_parts.get("coin_reward", 0.0)), 4),
                "death_penalty": round(float(reward_parts.get("death_penalty", 0.0)), 4),
                "finish_bonus": round(float(reward_parts.get("finish_bonus", 0.0)), 4),
                "dead": bool(info.get("dead", False)),
                "finished": bool(info.get("finished", False)),
                "model_path": str(model_path),
            }
            write_episode_row(csv_path, row)

            print(
                "Episode {episode}/{total} | reward={reward:.2f} | steps={steps} | "
                "epsilon={epsilon:.3f} | loss={loss:.4f} | x={x:.1f}".format(
                    episode=episode,
                    total=config.episodes,
                    reward=episode_reward,
                    steps=episode_step,
                    epsilon=row["epsilon"],
                    loss=avg_loss,
                    x=row["x_pos"],
                )
            )

        final_model_path = run_dir / "models" / "q_network_final.weights.h5"
        q_network.save_weights(final_model_path)
        print(f"Final model: {final_model_path}")
        print(f"Training log: {csv_path}")
    finally:
        env.close()


def parse_args():
    defaults = DQNConfig()
    parser = argparse.ArgumentParser(description="Train a DQN agent on the custom Mario Gymnasium env.")
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--frame-skip", type=int, default=defaults.frame_skip)
    parser.add_argument("--replay-capacity", type=int, default=defaults.replay_capacity)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--train-every", type=int, default=defaults.train_every)
    parser.add_argument("--target-update-every", type=int, default=defaults.target_update_every)
    parser.add_argument("--epsilon-start", type=float, default=defaults.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=defaults.epsilon_end)
    parser.add_argument("--epsilon-decay-steps", type=int, default=defaults.epsilon_decay_steps)
    parser.add_argument("--save-every", type=int, default=defaults.save_every)
    parser.add_argument("--level-name", default=defaults.level_name)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--run-name", default=defaults.run_name)
    parser.add_argument("--render-mode", choices=["rgb_array", "human"], default=defaults.render_mode)
    parser.add_argument("--fps", type=int, default=defaults.fps, help="FPS cap in human render mode. Use 0 for uncapped.")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser.parse_args()


def main():
    args = parse_args()
    config = DQNConfig(**vars(args))
    train(config)


if __name__ == "__main__":
    main()
