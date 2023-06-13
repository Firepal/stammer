from enum import Enum

class Side(Enum):
    LEFT = 1
    BOTTOM = 2
    RIGHT = 3
    TOP = 4

    @classmethod
    def next_ccw(cls, current):
        if current == cls.LEFT:
            return cls(2)
        if current == cls.BOTTOM:
            return cls(3)
        if current == cls.RIGHT:
            return cls(4)
        if current == cls.TOP:
            return cls(1)


class Tiling:
    def __init__(self, height: int, width: int) -> None:
        self.top = 0
        self.bottom = height
        self.left = 0
        self.right = width
        if width > height:
            self.side = Side.LEFT
        else:
            self.side = Side.BOTTOM
        self.needs_tiling = True
        
    def get_image_placement(self):
        pos_height, pos_width, pos_x0, pos_y0 = None, None, None, None
        if self.side in [Side.LEFT, Side.RIGHT]:
            pos_height = self.bottom - self.top
            pos_width = (self.right - self.left) // 2
            pos_y0 = self.top
            if self.side == Side.LEFT:
                pos_x0 = self.left
                self.left += pos_width
            else:
                self.right -= pos_width
                pos_x0 = self.right
        else:
            pos_width = self.right - self.left
            pos_height = (self.bottom - self.top) // 2
            pos_x0 = self.left
            if self.side == Side.TOP:
                pos_y0 = self.top
                self.top += pos_height
            else:
                self.bottom -= pos_height
                pos_y0 = self.bottom
        self.side = Side.next_ccw(self.side)
        self.needs_tiling = not self.needs_tiling
        return (pos_x0, pos_y0, pos_width, pos_height)
