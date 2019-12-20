import pygame as pg
import numpy as np

#############
# VARIABLES #
#############

# Window
win_side = 500
win = pg.display.set_mode((win_side, win_side))
pg.display.set_caption('A* pathfinding algorithm')

# Grid
grid = np.zeros((50, 50))

#############
# FUNCTIONS #
#############


def check_click():
    clicks = pg.mouse.get_pressed()

    if clicks[0]:
        x, y = pg.mouse.get_pos()

        row = int(y / 500 * 50)
        col = int(x / 500 * 50)

        grid[row, col] = -1


def draw_window():
    win.fill((255, 255, 255))

    for i in range(50):
        pg.draw.line(win, (0, 0, 0), (10 * i, 0), (10 * i, 500), 1)
        pg.draw.line(win, (0, 0, 0), (0, 10 * i), (500, 10 * i), 1)

    y_mul, x_mul = np.where(grid == -1)
    indexes = list(zip(x_mul, y_mul))

    for x, y in indexes:
        x *= 10
        y *= 10
        pg.draw.rect(win, (0, 0, 0), (x, y, 10, 10))

    pg.display.flip()


#############
# MAIN LOOP #
#############

clock = pg.time.Clock()
run = True

while run:
    clock.tick(30)

    for e in pg.event.get():
        if e.type == pg.QUIT:
            run = False

    check_click()
    draw_window()

pg.quit()
