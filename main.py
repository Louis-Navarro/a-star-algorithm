from algorithm import Algorithm

import pygame as pg
import numpy as np


class Window:
    BACKGROUND_COLOR = (255, 255, 255)
    SEARCHED = False

    def __init__(self, width=500, height=500, square=10, title='A* Algorithm', win=None, fps=60):
        """This class manages the pygame window and interactions with the window to draw the grid and use the algorithm.

        Args:
            width (int, optional): The width of the window (must be a multiple of `square`). Defaults to 500.
            height (int, optional): The height of the window (must be a multiple of `square`). Defaults to 500.
            square (int, optional): The size of a square in the grid. Defaults to 10.
            title (str, optional): The title of the window. Defaults to 'A* Algorithm'.
            win (pygame.Surface, optional): Initial window object. Defaults to None.
            fps (int, optional): FPS at which the window runs. Defaults to 60.
        """
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
        self.win = win if win else pg.display.set_mode(
            (width, height))  # Window
        pg.display.set_caption(title)  # Title
        print(type(self.win))

        self.font = pg.font.SysFont('sourcecodepro', 5)

        self.clock = pg.time.Clock()

    def check_quit(self):
        """Checks if the user wants to quit the application

        Returns:
            bool: Returns True if the user wants to quit the application, otherwise returns False
        """
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.run = False
                return True
        return False

    def draw_window(self, refresh):
        """The program calls this function to draw the grid with the starting and ending points, borders, and the path if there it was computed

        Args:
            refresh (bool): Wether or not the window should be refreshed (running pg.display.flip())
        """
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
            if (x > 0):
                x += 1
            if (y > 0):
                y += 1
            pg.draw.rect(self.win, 0, ((x, y), (self.square-1, self.square-1)))

        # Path
        inx = np.argwhere(self.grid == 3)
        for path in inx:
            y, x = path
            x, y = x*self.square, y*self.square
            if (x > 0):
                x += 1
            if (y > 0):
                y += 1
            pg.draw.rect(self.win, (0, 0, 255),
                         ((x, y), (self.square-1, self.square-1)))

        # Start
        y, x = self.start
        x, y = x*self.square, y*self.square
        if (x > 0):
            x += 1
        if (y > 0):
            y += 1
        pg.draw.rect(self.win, (0, 255, 0),
                     ((x, y), (self.square-1, self.square-1)))

        # End
        y, x = self.end
        x, y = x*self.square, y*self.square
        if (x > 0):
            x += 1
        if (y > 0):
            y += 1
        pg.draw.rect(self.win, (255, 0, 0),
                     ((x, y), (self.square-1, self.square-1)))

        if refresh:
            pg.display.flip()

    def check_click(self):
        """Function that checks if the user pressed any mouse button, and if so, updates the grid accordingly
        """
        pressed = pg.mouse.get_pressed()
        if any(pressed):
            position = pg.mouse.get_pos()
            row = position[1] // self.square
            col = position[0] // self.square

            val = 0
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
        """Function that check if the user pressed the escape key, and if so, clears the grid
        """
        pressed = pg.key.get_pressed()

        if not self.SEARCHED and pressed[pg.K_RETURN]:
            self.search()

        elif pressed[pg.K_ESCAPE]:
            self.grid = np.zeros(
                (self.width // self.square, self.height // self.square))
            self.SEARCHED = False

    def frame(self, quit=True, draw=True, click=True, keys=True):
        """Function that is called every frame, and is responsible for the frame-to-frame action

        Args:
            quit (bool, optional): Wether or not the program should check if the user wants to quit. Defaults to True.
            draw (bool, optional): Wether or not the program should draw the grid. Defaults to True.
            click (bool, optional): Wether or not the program should check if the user pressed his mouse. Defaults to True.
            keys (bool, optional): Wether or not the program should check if the user pressed the escape key on his keyboard. Defaults to True.

        Returns:
            int: Returns 0 if the user quit the program, otherwise does not return anything
        """
        if self.fps:
            self.clock.tick(self.fps)
        if quit:
            if self.check_quit():
                return 0
        if draw:
            self.draw_window(True)
        if click:
            self.check_click()
        if keys:
            self.check_keys()

    def search(self):
        """Function responsible for searching the optimal path between the points
        """
        self.SEARCHED = True
        self.algo.find_path(self.grid, self.start, self.end)


if __name__ == '__main__':
    win = Window(fps=None)
    while win.run:
        win.frame()
    pg.quit()
