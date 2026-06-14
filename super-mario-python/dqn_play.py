import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from dqn_train import append_frame, build_q_network, make_initial_state, preprocess_observation
from rl_env import SuperMarioPythonEnv


def load_run_config(run_dir):
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_model_path(run_dir, model_path):
    if model_path:
        return Path(model_path)

    final_path = run_dir / "models" / "q_network_final.weights.h5"
    if final_path.exists():
        return final_path

    snapshots = sorted((run_dir / "models").glob("q_network_episode_*.weights.h5"))
    if snapshots:
        return snapshots[-1]

    raise FileNotFoundError(f"No .weights.h5 model found in {run_dir / 'models'}")


def choose_action(q_network, state, env, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()

    q_values = q_network(np.expand_dims(state, axis=0), training=False)
    return int(tf.argmax(q_values[0]).numpy())


def play(args):
    run_dir = Path(args.run_dir)
    run_config = load_run_config(run_dir)
    model_path = resolve_model_path(run_dir, args.model_path)

    image_size = args.image_size or int(run_config.get("image_size", 84))
    stack_size = args.stack_size or int(run_config.get("stack_size", 4))
    frame_skip = args.frame_skip or int(run_config.get("frame_skip", 4))
    level_name = args.level_name or run_config.get("level_name", "Level1-1")

    env = SuperMarioPythonEnv(
        level_name=level_name,
        render_mode=args.render_mode,
        frame_skip=frame_skip,
        max_steps=args.max_steps,
        fps=args.fps,
    )

    input_shape = (image_size, image_size, stack_size)
    q_network = build_q_network(input_shape, env.action_space.n)
    q_network.load_weights(model_path)

    print(f"Loaded model: {model_path}")
    print(f"Actions: {env.get_action_meanings()}")

    try:
        for episode in range(1, args.episodes + 1):
            obs, info = env.reset()
            first_frame = preprocess_observation(obs, image_size)
            frame_stack, state = make_initial_state(first_frame, stack_size)
            episode_reward = 0.0

            for step in range(1, args.max_steps + 1):
                action = choose_action(q_network, state, env, args.epsilon)
                obs, reward, terminated, truncated, info = env.step(action)

                frame = preprocess_observation(obs, image_size)
                state = append_frame(frame_stack, frame)
                episode_reward += reward

                if args.print_every and (step == 1 or step % args.print_every == 0 or terminated or truncated):
                    print(
                        f"Episode {episode}, Step {step}, "
                        f"Action {info['action_name']}, Reward {reward:.2f}, "
                        f"Total {episode_reward:.2f}, X {info['x_pos']:.1f}"
                    )

                if terminated or truncated:
                    break

            print(
                f"Episode {episode} finished: reward={episode_reward:.2f}, "
                f"steps={step}, x={info.get('x_pos', 0):.1f}, "
                f"dead={info.get('dead')}, finished={info.get('finished')}"
            )
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Play Mario with a trained DQN weights file.")
    parser.add_argument("run_dir", help="Training run folder, e.g. training_runs/dqn_20260611_092756")
    parser.add_argument("--model-path", default="", help="Optional explicit .weights.h5 path.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--epsilon", type=float, default=0.0, help="Exploration while playing. 0.0 means greedy.")
    parser.add_argument("--render-mode", choices=["human", "rgb_array"], default="human")
    parser.add_argument("--fps", type=int, default=60, help="FPS cap in human render mode. Use 0 for uncapped.")
    parser.add_argument("--level-name", default="")
    parser.add_argument("--frame-skip", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--stack-size", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    play(args)


if __name__ == "__main__":
    main()
