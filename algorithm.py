from math import sqrt

import numpy as np


class Algorithm:
    INF = float('INF')

    def __init__(self, shape, diag=True):
        """The A* Algorithm is a path-finding algorithm, considered to be the most effective.
        To use this implimentation, initialize the class with the dimensions (n*m) of the grid, and if the path can go in diagonal.
        Then, run find_path() to find the optimal path.

        Args:
            shape (tuple): The shape of the grid
            diag (bool, optional): Wether or not the algorithm can go in diagonal. Defaults to True.
        """

        self.shape = shape

        # Keep track of visited nodes
        self.vis = np.full(shape, False)
        # Distances of nodes to visit
        self.dis = np.full(shape, np.inf)
        self.parent = {}

        if diag:  # Eight neighbors
            self.dX = (-1, -1, 0, 1, 1, 1, 0, -1)
            self.dY = (0, 1, 1, 1, 0, -1, -1, -1)
        else:  # Four neighbors
            self.dX = (-1, 0, 1, 0)
            self.dY = (0, -1, 0, 1)

    def find_path(self, grid, beg, end):
        """Call this function to find the optimal path between the start point and the end point.
        BE AWARE: THIS FUNCTION OVERWRITES THE GIVEN GRID, SO YOU MAY WANT TO CONSIDER COPYING THE GRID BEFORE CALLING THIS FUNCTION

        Args:
            grid (ndarray): Array representing the grid, containing a start point (1), an end point (2) and possibly walls (-1)
            beg (Iterable): Iterable containing the coordinates of the starting point
            end (Iterable): Iterable containing the coordinates of the ending point

        Returns:
            bool: Returns False if the algorithm was unable to find a path between the starting and ending points, otherwise returns True
        """

        print("Exploring ...")
        if self.explore(grid, beg, end) == False:
            return False
        print("Backtracking ...")
        self.backtrack(grid, end, beg)

        # Reinitialize variables
        self.vis = np.full(self.shape, False)
        self.dis = np.full(self.shape, np.inf)
        self.parent = {}

        return True

    def ok(self, a, b):
        """Check if a point of coordinate a and b is in the grid

        Args:
            a (int): Coordinate on the y-axis (line number)
            b (int): Coordinate on the x-axis (column number)

        Returns:
            bool: Returns `true` if the point is in the grid, otherwise return `false`
        """
        return a >= 0 and a < self.shape[0] and b >= 0 and b < self.shape[1]

    @staticmethod
    def get_distance(cur, dest):
        """Computes the distance between two points of the grid

        Args:
            cur (Iterable): List/tuple/ndarray representing the coordinates of the first point
            dest (Iterable): List/tuple/ndarray representing the coordinates of the second point

        Returns:
            float: Returns the distance between the two points as a decimal number
        """
        ans = sqrt(abs(cur[0]-dest[0])**2 + abs(cur[1]-dest[1])**2)
        return ans

    def explore(self, grid, beg, end):
        """Explores the grid and computes the score for each point between the starting and ending points.
        These scores are then used to find the optimal paths between the two points.

        Args:
            grid (ndarray): Array representing the grid, containing a starting (1) and ending (2) points, and possibly walls (-1)
            beg (Iterable): List/tuple/ndarray representing the coordinates of the first point
            end (Iterable): List/tuple/ndarray representing the coordinates of the second point

        Returns:
            bool: Returns False if the algorithm was unable to find a path between the starting and ending points, otherwise returns True
        """
        self.dis[beg[0], beg[1]] = self.get_distance(beg, end)

        cur = beg
        while cur != end:
            self.vis[cur[0], cur[1]] = True
            p_dis = self.dis[cur[0], cur[1]]
            self.dis[cur[0], cur[1]] = np.inf

            for k in range(len(self.dX)):
                i = self.dX[k]
                j = self.dY[k]
                x, y = cur[0]+i, cur[1]+j
                if self.ok(x, y) and self.dis[x, y] == np.inf and self.vis[x, y] == False and grid[x, y] != -1:
                    cur_dis = round(p_dis + self.get_distance(cur, (x, y)), 2)
                    if cur_dis < self.dis[x, y]:
                        self.dis[x, y] = cur_dis
                        self.parent[(x, y)] = cur

            cur = np.unravel_index(np.argmin(self.dis, axis=None), self.shape)
            if np.min(self.dis, axis=None) == np.inf:
                return False
        return True

    def backtrack(self, grid, beg, end):
        """Finds the optimal path between the points

        Args:
            grid (ndarray): Array representing the grid, containing a starting (1) and ending (2) points, and possibly walls (-1)
            beg (Iterable): List/tuple/ndarray representing the coordinates of the first point
            end (Iterable): List/tuple/ndarray representing the coordinates of the second point
        """
        cur = beg
        while cur != end:
            grid[cur[0], cur[1]] = 3
            cur = self.parent[cur]
        grid[cur[0], cur[1]] = 3


if __name__ == '__main__':
    import sys

    np.set_printoptions(threshold=sys.maxsize)

    grid = np.zeros((20, 20))

    x1, y1 = np.random.randint(0, 20, 2)
    grid[x1, y1] = 1

    x2, y2 = np.random.randint(0, 20, 2)
    grid[x2, y2] = 2

    print('Initial grid :')
    print(grid)

    algo = Algorithm((20, 20))
    algo.find_path(grid, (x1, y1), (x2, y2))

    print('End grid :')
    print(grid)
