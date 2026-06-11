import argparse
import time

import mo_gymnasium as mo_gym


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MO-SuperMario with random actions, without reinforcement learning."
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play.")
    parser.add_argument("--max-steps", type=int, default=5000, help="Maximum steps per episode.")
    parser.add_argument("--fps", type=float, default=30.0, help="Render speed when using human rendering.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Use rgb_array mode instead of opening a game window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    render_mode = "rgb_array" if args.no_render else "human"
    sleep_time = 0.0 if args.no_render or args.fps <= 0 else 1.0 / args.fps

    env = mo_gym.make("mo-supermario-v0", render_mode=render_mode)

    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    print("Random policy: each step uses env.action_space.sample()")
    print()

    try:
        for episode in range(1, args.episodes + 1):
            obs, info = env.reset()
            total_reward = None

            for step in range(1, args.max_steps + 1):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                if total_reward is None:
                    total_reward = reward.copy()
                else:
                    total_reward += reward

                if not args.no_render:
                    env.render()
                    time.sleep(sleep_time)

                if step == 1 or step % 50 == 0 or terminated or truncated:
                    print(
                        f"Episode {episode}, Step {step}, "
                        f"Action {action}, Reward {reward}, Total {total_reward}"
                    )

                if terminated or truncated:
                    reason = "terminated" if terminated else "truncated"
                    print(f"Episode {episode} ended after {step} steps ({reason}).")
                    print()
                    break
            else:
                print(f"Episode {episode} reached max_steps={args.max_steps}.")
                print()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
