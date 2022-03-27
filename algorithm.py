from math import sqrt


import numpy as np


class Algorithm:
    INF = float('INF')

    def __init__(self, shape, diag=True):
        self.shape = shape

        self.vis = np.full(shape, False)           # Keep track of visited nodes
        self.dis = np.full(shape, np.inf)          # Distances of nodes to visit
        self.parent = {}

        if diag: # Eight neighbors
            self.dX = (-1, -1, 0, 1, 1, 1, 0, -1)
            self.dY = (0, 1, 1, 1, 0, -1, -1, -1)
        else: # Four neighbors
            self.dX = (-1, 0, 1, 0)
            self.dY = (0, -1, 0, 1)

    def find_path(self, grid, beg, end):
        print("Exploring ...")
        self.explore(grid, beg, end)
        print("Backtracking ...")
        self.backtrack(grid, end, beg)
        self.__init__(self.shape)

    def ok(self, a, b):
        return a>=0 and a<self.shape[0] and b>=0 and b<self.shape[1]

    @staticmethod
    def get_distance(cur, dest):
        ans=sqrt(abs(cur[0]-dest[0])**2 + abs(cur[1]-dest[1])**2)
        return ans

    def explore(self, grid, beg, end):
        self.dis[beg[0], beg[1]] = self.get_distance(beg, end)

        cur=beg
        while cur!=end:
            self.vis[cur[0], cur[1]] = True
            p_dis=self.dis[cur[0], cur[1]] 
            self.dis[cur[0], cur[1]] = np.inf

            for k in range(len(self.dX)):
                i=self.dX[k]
                j=self.dY[k]
                x, y = cur[0]+i, cur[1]+j
                if self.ok(x, y) and self.dis[x, y]==np.inf and self.vis[x, y]==False and grid[x, y] != -1:
                    cur_dis = round(p_dis + self.get_distance(cur, (x, y)), 2)
                    if cur_dis < self.dis[x, y]:
                        self.dis[x, y] = cur_dis
                        self.parent[(x, y)] = cur
            
            cur = np.unravel_index(np.argmin(self.dis, axis=None), self.shape)

    def backtrack(self, grid, beg, end):
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
    