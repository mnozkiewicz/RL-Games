from .utils import Action


G = 1.0
T = 0.1
FLAP_STRENGTH = -0.3

INIT_X = 0.3
INIT_Y = 0.5
BIRD_SIZE = 0.06


class Bird:
    def __init__(self):
        self.x = INIT_X

        self.y = INIT_Y
        self.y_speed = 0.0
        self.size = BIRD_SIZE

    def reset(self):
        self.x = INIT_X
        self.y = INIT_Y
        self.y_speed = 0.0

    def step(self, action: Action):
        if action == Action.NOTHING:
            self.y_speed += G * T
        else:
            self.y_speed = FLAP_STRENGTH

        self.y += self.y_speed * T
