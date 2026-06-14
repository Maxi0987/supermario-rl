import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pygame


WINDOW_SIZE = (640, 480)


class _NullChannel:
    def play(self, *args, **kwargs):
        return None

    def stop(self):
        return None

    def get_busy(self):
        return False


class SilentSound:
    def __init__(self):
        self.allowSFX = False
        self.music_channel = _NullChannel()
        self.sfx_channel = _NullChannel()
        self.soundtrack = None
        self.coin = None
        self.bump = None
        self.stomp = None
        self.jump = None
        self.death = None
        self.kick = None
        self.brick_bump = None
        self.powerup = None
        self.powerup_appear = None
        self.pipe = None

    def play_sfx(self, sfx):
        return None

    def play_music(self, music):
        return None


@contextmanager
def project_cwd():
    previous = Path.cwd()
    os.chdir(Path(__file__).resolve().parent)
    try:
        yield
    finally:
        os.chdir(previous)


class MarioGame:
    def __init__(
        self,
        level_name="Level1-1",
        render_mode="rgb_array",
        window_size=WINDOW_SIZE,
        fps=60,
        sound=False,
    ):
        self.level_name = level_name
        self.render_mode = render_mode
        self.window_size = window_size
        self.fps = fps
        self.sound_enabled = sound
        self.screen = None
        self.clock = None
        self.dashboard = None
        self.sound = None
        self.level = None
        self.mario = None
        self.last_reward_parts = {}

        self._init_pygame()
        self.reset()

    def _init_pygame(self):
        if self.render_mode != "human":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        if not self.sound_enabled:
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

        if self.sound_enabled:
            pygame.mixer.pre_init(44100, -16, 2, 4096)
        pygame.init()

        self.screen = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()

    def reset(self):
        with project_cwd():
            from classes.Dashboard import Dashboard
            from classes.Level import Level
            from classes.Sound import Sound
            from entities.Mario import Mario

            self.dashboard = Dashboard("./img/font.png", 8, self.screen)
            self.dashboard.state = "start"
            self.dashboard.time = 0
            self.dashboard.levelName = self.level_name.split("Level")[-1]

            self.sound = Sound() if self.sound_enabled else SilentSound()
            self.level = Level(self.screen, self.sound, self.dashboard)
            self.level.loadLevel(self.level_name)
            self.mario = Mario(
                0,
                0,
                self.level,
                self.screen,
                self.dashboard,
                self.sound,
                enable_input=False,
                terminal_animation=False,
            )

        self.draw()
        return self.get_observation()

    def step(self, action):
        pygame.event.pump()

        previous_x = self.x_pos
        previous_score = self.dashboard.points
        previous_coins = self.dashboard.coins

        self.draw()
        self.mario.update(action=action, process_input=False)

        terminated = self.is_dead or self.is_finished
        reward = self._reward(previous_x, previous_score, previous_coins, terminated)
        info = self.info()
        info["reward_parts"] = self.last_reward_parts

        if self.render_mode == "human":
            pygame.display.set_caption("RL Super Mario ({:d} FPS)".format(int(self.clock.get_fps())))
            pygame.display.update()
            if self.fps:
                self.clock.tick(self.fps)

        return self.get_observation(), reward, terminated, info

    def draw(self):
        self.level.drawLevel(self.mario.camera)
        self.dashboard.update()

    def get_observation(self):
        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2)).astype(np.uint8)

    def render(self):
        if self.render_mode == "human":
            pygame.display.update()
            return None
        return self.get_observation()

    def close(self):
        pygame.quit()

    def _reward(self, previous_x, previous_score, previous_coins, terminated):
        x_delta = self.x_pos - previous_x
        x_reward = max(0.0, x_delta) * 2.0
        score_reward = (self.dashboard.points - previous_score) * 0.001
        coin_reward = (self.dashboard.coins - previous_coins) * 5.0
        time_penalty = -0.01
        death_penalty = -25.0 if self.is_dead else 0.0
        finish_bonus = 100.0 if self.is_finished and terminated else 0.0
        total_reward = x_reward + score_reward + coin_reward + time_penalty + death_penalty + finish_bonus

        self.last_reward_parts = {
            "x_delta": float(x_delta),
            "x_reward": float(x_reward),
            "score_reward": float(score_reward),
            "coin_reward": float(coin_reward),
            "time_penalty": float(time_penalty),
            "death_penalty": float(death_penalty),
            "finish_bonus": float(finish_bonus),
            "total_reward": float(total_reward),
        }
        return float(total_reward)

    @property
    def x_pos(self):
        return float(self.mario.rect.x)

    @property
    def is_dead(self):
        return bool(self.mario.restart)

    @property
    def is_finished(self):
        return self.mario.getPosIndexAsFloat().x >= self.level.levelLength - 1

    def info(self):
        return {
            "x_pos": self.x_pos,
            "y_pos": float(self.mario.rect.y),
            "score": int(self.dashboard.points),
            "coins": int(self.dashboard.coins),
            "time": int(self.dashboard.time),
            "level": self.level_name,
            "power_up_state": int(self.mario.powerUpState),
            "dead": self.is_dead,
            "finished": self.is_finished,
        }
