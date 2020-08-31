from math import sqrt


import numpy as np


class Algorithm:

    # Four neighbors
    # dX = (-1, 0, 1, 0)
    # dY = (0, -1, 0, 1)

    # Eight neighbors
    dX = (-1, -1, 0, 1, 1, 1, 0, -1)
    dY = (0, 1, 1, 1, 0, -1, -1, -1)
    
    INF = float('INF')

    def __init__(self, shape):
        self.shape = shape

        self.vis = np.full(shape, False)           # Keep track of visited nodes
        self.dis = np.full(shape, np.inf)          # Distances of nodes to visit
        self.parent = {}

    def find_path(self, grid, beg, end):
        print("Exploring ...")
        self.explore(grid, beg, end)
        # print('Finished exploring')
        # print(self.dis)
        print("Backtracking ...")
        self.backtrack(grid, end, beg)
        # print('Finished backtracking')
        self.__init__(self.shape)

    def ok(self, a, b):
        return a>=0 and a<self.shape[0] and b>=0 and b<self.shape[1]

    @staticmethod
    def get_distance(cur, dest):
        ans=sqrt(abs(cur[0]-dest[0])**2 + abs(cur[1]-dest[1])**2)
        return ans

    def explore(self, grid, beg, end):
        # print(grid.shape, self.shape)
        self.dis[beg[0], beg[1]] = self.get_distance(beg, end)
        # iX, iY = beg
        # eX, eY = end

        cur=beg
        while cur!=end:
            self.vis[cur[0], cur[1]] = True
            p_dis=self.dis[cur[0], cur[1]] 
            self.dis[cur[0], cur[1]] = np.inf

            for k in range(len(self.dX)):
                i=self.dX[k]
                j=self.dY[k]
                x, y = cur[0]+i, cur[1]+j
                # if self.ok(x, y) and self.vis[x, y]==np.inf and self.dis[x, y]==np.inf and grid[x, y] != -1:
                if self.ok(x, y) and self.dis[x, y]==np.inf and self.vis[x, y]==False and grid[x, y] != -1:
                    # cur_dis = self.get_distance((x, y), beg)
                    # cur_dis += self.get_distance((x, y), end)
                    # cur_dis = round(cur_dis, 2)
                    # self.dis[x, y] = cur_dis
                    # self.parent[(x, y)] = cur
                    cur_dis = round(p_dis + self.get_distance(cur, (x, y)), 2)
                    if cur_dis < self.dis[x, y]:
                        self.dis[x, y] = cur_dis
                        self.parent[(x, y)] = cur
            
            cur = np.unravel_index(np.argmin(self.dis, axis=None), self.shape)

        # self.vis[cur[0], cur[1]] = self.dis[cur[0], cur[1]]
        # self.dis[cur[0], cur[1]] = np.inf

    def backtrack(self, grid, beg, end):
        cur = beg
        while cur != end:
            # self.vis[cur[0], cur[1]] = np.inf
            # grid[cur[0], cur[1]] = 3
            # mn = self.INF
            # # print(cur)
            # for k in range(4):
                # i=self.dX[k]
                # j=self.dY[k]
                # # print(i, j, end=' ')
                # x, y = cur[0]+i, cur[1]+j
                # # print(x, y)
                # # if mn>self.dis[x, y]>-1:
                    # # mn = self.dis[x, y]
                    # # cur = (x, y)
                    # # self.vis[x, y] = 0
                # if self.ok(x, y) and mn>self.vis[x, y]:
                    # mn=self.vis[x, y]
                    # new_cur=(x, y)
            # # print()
            # cur=new_cur
            # grid[cur[0], cur[1]] = 3
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
    
    # import pdb; pdb.set_trace()

    # pos = [(x, y) for x in range(20) for y in range(20)]
    # i=1

    # for x in pos:
        # for y in pos:
            # if (x==y): continue;
            # grid=np.zeros((20, 20))
            # print(i, x, y)
            # algo.find_path(grid, x, y)
            # i+=1
