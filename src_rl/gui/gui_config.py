from dataclasses import dataclass


@dataclass(frozen=True)
class GUIConfig:
    """
    Basic parameters for a GUI window.
    Includes height of the window, width and also the frame rate (per minute)
    """

    pixel_height: int = 600
    pixel_width: int = 600
    frame_rate: int = 10

    def __post_init__(self):
        if self.pixel_height <= 0 or self.pixel_width <= 0:
            raise ValueError("Pixel dimensions must be positive.")

        if self.pixel_height > 1600 or self.pixel_width > 1600:
            raise ValueError("Pixel dimensions must be at most 1600 pixels")

        if not (5 <= self.frame_rate <= 120):
            raise ValueError("Frame rate should be in range [5, 120]")
