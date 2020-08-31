from algorithm import Algorithm

import pygame as pg
import numpy as np

class Window:
    BACKGROUND_COLOR = (255, 255, 255)
    SEARCHED = False

    def __init__(self, width=500, height=500, square=10, title='A* Algorithm', win=None, fps=60):
        # General variables
        self.width = width
        self.height = height
        self.square = square
        self.fps = fps
        self.run = True

        self.start = 0, 0
        self.end = height-1, width-1

        # Algorithm
        self.algo = Algorithm((height//square, width//square))

        # Grid
        self.grid = np.zeros((width // square, height // square))

        # Pygame stuff
        pg.init()
        self.win = win if win else pg.display.set_mode((width, height)) # Window
        pg.display.set_caption(title) # Title

        self.font = pg.font.SysFont('sourcecodepro', 5)

        self.clock = pg.time.Clock()

    def check_quit(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.run = False
                return True
        return False

    def draw_window(self, refresh):
        self.win.fill(self.BACKGROUND_COLOR)

        # Wall lines
        for i in range(1, self.width // self.square):
            x = i * self.square
            pg.draw.line(self.win, 0, (x, 0), (x, self.height))

        for i in range(1, self.height // self.square):
            y = i * self.square
            pg.draw.line(self.win, 0, (0, y), (self.width, y))

        # Walls
        inx = np.argwhere(self.grid == -1)
        for wall in inx:
            y, x = wall
            x, y = x*self.square, y*self.square
            if (x>0): x+=1
            if (y>0): y+=1
            pg.draw.rect(self.win, 0, ((x, y), (self.square-1, self.square-1)))

        # Path
        inx = np.argwhere(self.grid == 3)
        for path in inx:
            y, x = path
            x, y = x*self.square, y*self.square
            if (x>0): x+=1
            if (y>0): y+=1
            pg.draw.rect(self.win, (0, 0, 255), ((x, y), (self.square-1, self.square-1)))

        # Start
        y, x = self.start
        x, y = x*self.square, y*self.square
        if (x>0): x+=1
        if (y>0): y+=1
        pg.draw.rect(self.win, (0, 255, 0), ((x, y), (self.square-1, self.square-1)))

        # End
        y, x = self.end
        x, y = x*self.square, y*self.square
        if (x>0): x+=1
        if (y>0): y+=1
        pg.draw.rect(self.win, (255, 0, 0), ((x, y), (self.square-1, self.square-1)))

        if refresh:
            pg.display.flip()

    def check_click(self):
        pressed = pg.mouse.get_pressed()
        if any(pressed):
            position = pg.mouse.get_pos()
            row = position[1] // self.square
            col = position[0] // self.square

            val=0
            # Left click = start
            if pressed[0]:
                self.start = row, col
                val = 1

            # Middle click = wall
            elif pressed[1]:
                self.grid[row, col] = -1

            # Right click = end
            elif pressed[2]:
                self.end = row, col


    def check_keys(self):
        pressed = pg.key.get_pressed()
        
        if not self.SEARCHED and pressed[pg.K_RETURN]:
            self.search()

        elif pressed[pg.K_ESCAPE]:
            self.grid = np.zeros((self.width // self.square, self.height // self.square))
            self.SEARCHED = False

    def frame(self, quit=True, draw=True, click=True, keys=True):
        if self.fps:
            self.clock.tick(self.fps)
        if quit:
            if self.check_quit(): return 0
        if draw:
            self.draw_window(True)
        if click:
            self.check_click()
        if keys:
            self.check_keys()

    def search(self):
        self.SEARCHED = True
        self.algo.find_path(self.grid, self.start, self.end)


if __name__ == '__main__':
    win = Window(fps=None)
    while win.run:
        win.frame()
    pg.quit()
