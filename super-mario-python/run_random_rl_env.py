import argparse

from rl_env import SuperMarioPythonEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Run the RL-ready Mario env with random actions.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--fps", type=int, default=60, help="FPS cap in human mode. Use 0 for uncapped.")
    parser.add_argument("--level", default="Level1-1")
    parser.add_argument("--human", action="store_true", help="Open a pygame window.")
    return parser.parse_args()


def main():
    args = parse_args()
    env = SuperMarioPythonEnv(
        level_name=args.level,
        render_mode="human" if args.human else "rgb_array",
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
        fps=args.fps,
    )

    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    print("Actions:", env.get_action_meanings())
    print()

    try:
        for episode in range(1, args.episodes + 1):
            obs, info = env.reset()
            total_reward = 0.0

            for step in range(1, args.max_steps + 1):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if step == 1 or step % 25 == 0 or terminated or truncated:
                    print(
                        f"Episode {episode}, Step {step}, "
                        f"Action {info['action_name']}, Reward {reward:.2f}, "
                        f"Total {total_reward:.2f}, X {info['x_pos']:.1f}"
                    )

                if terminated or truncated:
                    print(f"Episode {episode} ended: {info}")
                    print()
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()
