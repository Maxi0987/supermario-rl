from classes.Maths import Vec2D


class Camera:
    def __init__(self, pos, entity):
        self.pos = Vec2D(pos.x, pos.y)
        self.entity = entity
        self.x = self.pos.x * 32
        self.y = self.pos.y * 32

    def move(self):
        xPosFloat = self.entity.getPosIndexAsFloat().x
        right_scroll_limit = max(10, self.entity.levelObj.levelLength - 10)
        if 10 < xPosFloat < right_scroll_limit:
            self.pos.x = -xPosFloat + 10
        self.x = self.pos.x * 32
        self.y = self.pos.y * 32
