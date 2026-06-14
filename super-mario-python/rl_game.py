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
        self.last_action_index = -1
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

        self.last_action_index = -1
        self.draw()
        return self.get_observation()

    def step(self, action):
        pygame.event.pump()
        self.last_action_index = int(action.get("index", -1))

        previous_x = self.x_pos
        previous_score = self.dashboard.points
        previous_enemy_points = self.mario.enemy_points
        previous_coins = self.dashboard.coins

        self.draw()
        self.mario.update(action=action, process_input=False)

        terminated = self.is_dead or self.is_finished
        reward = self._reward(previous_x, previous_score, previous_enemy_points, previous_coins, terminated)
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

    def get_frame(self):
        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2)).astype(np.uint8)

    def get_observation(self):
        features = []
        level_width = max(1.0, float(self.level.levelLength * 32))
        level_height = max(1.0, float(len(self.level.level) * 32))

        features.extend(
            [
                self._clip(self.mario.rect.x / level_width),
                self._clip(self.mario.rect.y / level_height),
                self._clip(self.mario.vel.x / 5.0),
                self._clip(self.mario.vel.y / 15.0),
                1.0 if self.mario.onGround else 0.0,
                1.0 if self.mario.inJump else 0.0,
                self._clip(float(self.mario.powerUpState)),
                self._clip((level_width - self.mario.rect.x) / level_width),
            ]
        )

        features.extend(self._last_action_features())
        features.extend(self._local_tile_features())
        features.extend(self._nearest_mob_features())

        return np.asarray(features, dtype=np.float32)

    def render(self):
        if self.render_mode == "human":
            pygame.display.update()
            return None
        return self.get_frame()

    def close(self):
        pygame.quit()

    def _reward(self, previous_x, previous_score, previous_enemy_points, previous_coins, terminated):
        x_delta = self.x_pos - previous_x
        x_reward = max(0.0, x_delta) * 2.0
        score_delta = self.dashboard.points - previous_score
        enemy_score_delta = self.mario.enemy_points - previous_enemy_points
        non_enemy_score_delta = max(0, score_delta - enemy_score_delta)
        score_reward = non_enemy_score_delta * 0.001
        coin_reward = (self.dashboard.coins - previous_coins) * 5.0
        time_penalty = -0.1
        death_penalty = -25.0 if self.is_dead else 0.0
        finish_bonus = 100.0 if self.is_finished and terminated else 0.0
        total_reward = x_reward + score_reward + coin_reward + time_penalty + death_penalty + finish_bonus

        self.last_reward_parts = {
            "x_delta": float(x_delta),
            "x_reward": float(x_reward),
            "score_reward": float(score_reward),
            "enemy_score_ignored": float(enemy_score_delta),
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

    def _last_action_features(self):
        features = [0.0] * 8
        if 0 <= self.last_action_index < len(features):
            features[self.last_action_index] = 1.0
        return features

    def _local_tile_features(self):
        mario_tile_x = self.mario.rect.x // 32
        mario_tile_y = self.mario.rect.y // 32
        features = []

        for y_offset in (-2, -1, 0, 1):
            for x_offset in (-1, 0, 1, 2, 3):
                x = int(mario_tile_x + x_offset)
                y = int(mario_tile_y + y_offset)
                features.append(1.0 if self._is_solid_tile(x, y) else 0.0)

        return features

    def _is_solid_tile(self, x, y):
        if x < 0 or x >= self.level.levelLength:
            return True
        if y < 0:
            return False
        if y >= len(self.level.level):
            return True
        return self.level.level[y][x].rect is not None

    def _nearest_mob_features(self):
        mobs = [
            entity
            for entity in self.level.entityList
            if entity.type == "Mob" and entity.alive is not None
        ]
        mobs.sort(key=lambda entity: abs(entity.rect.x - self.mario.rect.x))

        features = []
        for mob in mobs[:2]:
            dx = (mob.rect.x - self.mario.rect.x) / (10.0 * 32)
            dy = (mob.rect.y - self.mario.rect.y) / (5.0 * 32)
            features.extend(
                [
                    self._clip(dx),
                    self._clip(dy),
                    self._clip(mob.vel.x / 5.0),
                    1.0 if mob.alive else 0.0,
                    1.0 if mob.active else 0.0,
                ]
            )

        while len(features) < 10:
            features.append(0.0)

        return features

    @staticmethod
    def _clip(value):
        return float(np.clip(value, -1.0, 1.0))
