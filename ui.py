"""Pygame helpers: asset loading, sprites, buttons, etc."""

from __future__ import annotations

import pygame as pg
from importlib.resources import files
from pathlib import Path
from typing import cast

# ---------------------------------------------------------------------------
# Asset folders                                                              |
# ---------------------------------------------------------------------------

_ASSETS: Path = cast(Path, files(__package__) / "assets")
_IMG_DIR: Path = _ASSETS / "images"
_SND_DIR: Path = _ASSETS / "sounds"

# ---------------------------------------------------------------------------
# Loader helpers                                                             |
# ---------------------------------------------------------------------------


def load_image(name: str) -> pg.Surface:
    """
    Return a Pygame :class:`Surface` loaded from **assets/images/**.

    The extra :pyfunc:`typing.cast` keeps static type-checkers happy
    (pygame’s stub metadata is still incomplete on Py 3.12 / 3.13).
    """
    path: Path = _IMG_DIR / name
    return cast(pg.Surface, pg.image.load(path)).convert_alpha()


def load_sound(name: str) -> pg.mixer.Sound:
    """
    Return a Pygame :class:`Sound` loaded from **assets/sounds/**.
    """
    path: Path = _SND_DIR / name
    return cast(pg.mixer.Sound, pg.mixer.Sound(path))


# ---------------------------------------------------------------------------
# Simple clickable text button                                               |
# ---------------------------------------------------------------------------


class Button:
    """A one-off rectangular text button."""

    def __init__(
        self,
        label: str,
        pos: tuple[int, int],
        *,
        colour: tuple[int, int, int] = (255, 0, 0),
        font_name: str = "Times New Roman",
        font_size: int = 36,
    ) -> None:
        font = pg.font.SysFont(font_name, font_size)
        self.surface: pg.Surface = font.render(label, True, colour)
        self.rect: pg.Rect = self.surface.get_rect(topleft=pos)

    # ---------------------------------------------------------------------

    def draw(self, screen: pg.Surface) -> None:
        """Blit the button once per frame."""
        screen.blit(self.surface, self.rect)

    def clicked(self, mouse_pos: tuple[int, int]) -> bool:
        """Return ``True`` if *mouse_pos* is inside the button rect."""
        return self.rect.collidepoint(mouse_pos)
