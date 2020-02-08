import algorithm
from threading import Thread

import pygame as pg
import numpy as np

pg.init()

#############
# VARIABLES #
#############

# Window
win_side = 500
win = pg.display.set_mode((win_side, win_side))
pg.display.set_caption('A* Algorithm')

# Grid
square_size = 10
grid = np.zeros((win_side // square_size, win_side // square_size))
values = grid.copy()

# Font
font = pg.font.SysFont('sourcecodepro', 5)

#############
# FUNCTIONS #
#############


def draw_window():
    win.fill((255, 255, 255))

    walls = np.argwhere(grid == -1)
    for wall in walls:
        x = wall[1] * square_size
        y = wall[0] * square_size
        pg.draw.rect(win, 0, (x, y, square_size, square_size))

    green = np.argwhere(grid == 1)
    if green.any():
        for square in green:
            pg.draw.rect(win, (0, 255, 0),
                         (square[1] * square_size, square[0] * square_size,
                          square_size, square_size))

    red = np.argwhere(grid == 2)
    if red.any():
        for square in red:
            pg.draw.rect(win, (255, 0, 0),
                         (square[1] * square_size, square[0] * square_size,
                          square_size, square_size))

    path = np.argwhere(grid == 3)
    if path.any():
        for square in path:
            pg.draw.rect(win, (0, 0, 255),
                         (square[1] * square_size, square[0] * square_size,
                          square_size, square_size))

    f_costs = np.argwhere(values != 0)
    if f_costs.any():
        for square in f_costs:
            cost = values[square[0], square[1]]
            text = font.render(f'{cost:.0f}', True, (0, 0, 0), (255, 255, 255))

            x = square[1] * square_size + 2
            y = square[0] * square_size

            win.blit(text, (x, y))

    for i in range(win_side // square_size):
        pg.draw.line(win, 0, (square_size * i, 0), (square_size * i, win_side))
        pg.draw.line(win, 0, (0, square_size * i), (win_side, square_size * i))

    pg.display.flip()


def check_click():
    pressed = pg.mouse.get_pressed()

    if any(pressed):
        pos = pg.mouse.get_pos()
        row = pos[1] // square_size
        col = pos[0] // square_size

        if pressed[0]:
            grid[row, col] = -1

        elif pressed[2]:
            if not np.argwhere(grid == 1).any():
                grid[row, col] = 1

            elif not np.argwhere(grid == 2).any():
                grid[row, col] = 2


def check_presses():
    global grid, values, pressed_enter

    pressed = pg.key.get_pressed()

    if not pressed_enter:
        if pressed[pg.K_RETURN]:
            f1 = Thread(target=algorithm.find_path_animated,
                        args=(grid,))
            f1.start()
            pressed_enter = True

    if pressed[pg.K_ESCAPE]:
        grid = np.zeros((win_side // square_size, win_side // square_size))
        values = grid.copy()

        pressed_enter = False


#############
# MAIN LOOP #
#############

if __name__ == '__main__':
    pressed_enter = False

    # clock = pg.time.Clock()
    run = True
    while run:
        # clock.tick(1)

        for e in pg.event.get():
            if e.type == pg.QUIT:
                run = False

        draw_window()
        check_click()
        check_presses()

    pg.quit()
