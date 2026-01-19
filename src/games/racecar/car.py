from typing import Tuple
import math


class Car:
    CAR_LENGTH: float = 0.04
    CAR_WIDTH: float = 0.02
    TURN_ANGLE: int = 9
    ACCELARATION: float = 0.002
    MAX_SPEED: float = 0.05

    def __init__(self, initial_x: float, initial_y: float, initial_angle: int) -> None:
        self.x = initial_x
        self.y = initial_y
        self.speed = 0.002

        self.angle = initial_angle

        self.vector = (1.0, 0.0)
        self.vector_update()

    def vector_update(self) -> None:
        x_dir = math.cos(math.radians(self.angle))
        y_dir = math.sin(math.radians(self.angle))
        self.vector = (x_dir, y_dir)

    def move(self, command: int) -> None:
        match command:
            case -1:
                pass
            case 0:
                self.speed += Car.ACCELARATION
                self.speed = min(self.speed, Car.MAX_SPEED)
            case 1:
                self.angle += Car.TURN_ANGLE
                self.angle = self.angle % 360
                self.vector_update()
            case 2:
                self.angle -= Car.TURN_ANGLE
                self.angle = self.angle % 360
                self.vector_update()
            case 3:
                self.speed -= Car.ACCELARATION
                self.speed = max(self.speed, 0.002)
            case _:
                raise ValueError(
                    f"Unknow command value passed to turn method: {command}"
                )

        x_dir, y_dir = self.vector

        self.x += x_dir * self.speed
        self.y -= y_dir * self.speed

    def pos(self) -> Tuple[float, float]:
        return self.x, self.y
