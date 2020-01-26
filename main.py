import pygame as pg
import numpy as np
import algorithm

#############
# VARIABLES #
#############

# Window
win_side = 500
win = pg.display.set_mode((win_side, win_side))
pg.display.set_caption('A* Algorithm')

# Grid
grid = np.zeros((win_side // 10, win_side // 10))

#############
# FUNCTIONS #
#############


def draw_window():
    win.fill((255, 255, 255))

    for i in range(50):
        pg.draw.line(win, 0, (10 * i, 0), (10 * i, win_side))
        pg.draw.line(win, 0, (0, 10 * i), (win_side, 10 * i))

    walls = np.argwhere(grid == -1)
    for wall in walls:
        x = wall[1] * 10
        y = wall[0] * 10
        pg.draw.rect(win, 0, (x, y, 10, 10))

    start = np.argwhere(grid == 1)
    if start.any():
        pg.draw.rect(win, (0, 255, 0),
                     (start[0, 1] * 10, start[0, 0] * 10, 10, 10))

    end = np.argwhere(grid == 2)
    if end.any():
        pg.draw.rect(win, (255, 0, 0),
                     (end[0, 1] * 10, end[0, 0] * 10, 10, 10))

    path = np.argwhere(grid == 3)
    if path.any():
        for square in path:
            pg.draw.rect(win, (0, 0, 255),
                         (square[1] * 10, square[0] * 10, 10, 10))

    pg.display.flip()


def check_click():
    pressed = pg.mouse.get_pressed()

    if any(pressed):
        pos = pg.mouse.get_pos()
        row = pos[1] // 10
        col = pos[0] // 10

        if pressed[0]:
            grid[row, col] = -1

        elif pressed[2]:
            if not np.argwhere(grid == 1).any():
                grid[row, col] = 1

            elif not np.argwhere(grid == 2).any():
                grid[row, col] = 2


def check_presses():
    pressed = pg.key.get_pressed()

    if pressed[pg.K_RETURN]:
        algorithm.find_path(grid, True)
        return True

    return False


#############
# MAIN LOOP #
#############

if __name__ == '__main__':
    pressed_enter = False

    # clock = pg.time.Clock()
    run = True
    while run:
        # clock.tick(60)

        for e in pg.event.get():
            if e.type == pg.QUIT:
                run = False

        draw_window()
        check_click()
        if not pressed_enter:
            pressed_enter = check_presses()

    pg.quit()
