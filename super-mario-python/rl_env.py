import gymnasium as gym
import numpy as np

from rl_game import MarioGame, WINDOW_SIZE


ACTION_MAP = [
    {"name": "NOOP", "direction": 0, "jump": False, "boost": False},
    {"name": "LEFT", "direction": -1, "jump": False, "boost": False},
    {"name": "RIGHT", "direction": 1, "jump": False, "boost": False},
    {"name": "JUMP", "direction": 0, "jump": True, "boost": False},
    {"name": "LEFT_JUMP", "direction": -1, "jump": True, "boost": False},
    {"name": "RIGHT_JUMP", "direction": 1, "jump": True, "boost": False},
    {"name": "RIGHT_RUN", "direction": 1, "jump": False, "boost": True},
    {"name": "RIGHT_RUN_JUMP", "direction": 1, "jump": True, "boost": True},
]


class SuperMarioPythonEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        level_name="Level1-1",
        render_mode="rgb_array",
        frame_skip=4,
        max_steps=5000,
        fps=0,
        sound=False,
    ):
        super().__init__()
        self.level_name = level_name
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.fps = fps
        self.sound = sound
        self.steps = 0
        self.game = None

        width, height = WINDOW_SIZE
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 3),
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Discrete(len(ACTION_MAP))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        if self.game is not None:
            self.game.close()

        options = options or {}
        level_name = options.get("level_name", self.level_name)
        self.game = MarioGame(
            level_name=level_name,
            render_mode=self.render_mode,
            fps=self.fps,
            sound=self.sound,
        )
        return self.game.get_observation(), self.game.info()

    def step(self, action):
        self.steps += 1
        action_dict = ACTION_MAP[int(action)]
        total_reward = 0.0
        terminated = False
        info = {}
        obs = None

        for _ in range(self.frame_skip):
            obs, reward, terminated, info = self.game.step(action_dict)
            total_reward += reward
            if terminated:
                break

        truncated = self.steps >= self.max_steps
        info["action_name"] = action_dict["name"]
        info["frame_skip"] = self.frame_skip

        return obs, float(total_reward), terminated, truncated, info

    def render(self):
        return self.game.render()

    def close(self):
        if self.game is not None:
            self.game.close()
            self.game = None

    def get_action_meanings(self):
        return [action["name"] for action in ACTION_MAP]
