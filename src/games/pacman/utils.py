from enum import Enum
# from typing import Optional


class Action(Enum):
    QUIT = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    NOTHING = 5

    # def to_dir(self) -> Optional[Dir]:
    #     match self:
    #         case Action.UP:
    #             return Dir.UP
    #         case Action.DOWN:
    #             return Dir.DOWN
    #         case Action.LEFT:
    #             return Dir.LEFT
    #         case Action.RIGHT:
    #             return Dir.RIGHT
    #         case _:
    #             return None
