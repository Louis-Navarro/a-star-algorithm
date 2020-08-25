from math import sqrt


import numpy as np


class Algorithm:

    dX = (-1, 0, 1, 0)
    dY = (0, -1, 0, 1)
    INF = float('INF')

    def __init__(self, shape):
        self.shape = shape

        self.vis = np.full(shape, np.inf)          # Keep track of visited nodes' distances
        self.dis = np.full(shape, np.inf)          # Distances of nodes to visit

    def find_path(self, grid, beg, end):
        print("Exploring ...")
        self.explore(grid, beg, end)
        print("Backtracking ...")
        self.backtrack(grid, end, beg)

    def ok(self, a, b):
        return a>=0 and a<self.shape[0] and b>=0 and b<self.shape[1]

    @staticmethod
    def get_distance(cur, dest):
        ans=sqrt(abs(cur[0]-dest[0])**2 + abs(cur[1]-dest[1])**2)
        return ans

    def explore(self, grid, beg, end):
        self.dis[beg[0], beg[1]] = 0
        iX, iY = beg
        eX, eY = end

        cur=beg
        while cur!=end:
            self.vis[cur[0], cur[1]] = self.dis[cur[0], cur[1]]
            self.dis[cur[0], cur[1]] = np.inf
            for k in range(4):
                i=self.dX[k]
                j=self.dY[k]
                x, y = cur[0]+i, cur[1]+j
                if self.ok(x, y) and self.vis[x, y]==np.inf and grid[x, y] != -1:
                    cur_dis = self.get_distance((x, y), beg)
                    cur_dis += self.get_distance((x, y), end)
                    cur_dis = round(cur_dis, 2)
                    self.dis[x, y] = cur_dis
            
            cur = np.unravel_index(np.argmin(self.dis, axis=None), self.shape)

        self.vis[cur[0], cur[1]] = self.dis[cur[0], cur[1]]
        self.dis[cur[0], cur[1]] = np.inf

    def backtrack(self, grid, beg, end):
        cur = beg
        while cur != end:
            self.vis[cur[0], cur[1]] = np.inf
            grid[cur[0], cur[1]] = 3
            mn = self.INF
            # print(cur)
            for k in range(4):
                i=self.dX[k]
                j=self.dY[k]
                # print(i, j, end=' ')
                x, y = cur[0]+i, cur[1]+j
                # print(x, y)
                # if mn>self.dis[x, y]>-1:
                    # mn = self.dis[x, y]
                    # cur = (x, y)
                    # self.vis[x, y] = 0
                if self.ok(x, y) and mn>self.vis[x, y]:
                    mn=self.vis[x, y]
                    new_cur=(x, y)
            # print()
            cur=new_cur
            grid[cur[0], cur[1]] = 3
        grid[cur[0], cur[1]] = 3


if __name__ == '__main__':
    grid = np.zeros((9, 9))
    x, y = np.random.randint(0, 10, 2)
    grid[x,y] = 1
    x, y = np.random.randint(0, 10, 2)
    grid[x,y] = 2

    algo = Algorithm()
