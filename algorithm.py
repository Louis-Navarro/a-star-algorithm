import json

import numpy as np


def reveal_surrounding_nodes(node, start, end, grid, OPEN, CLOSED, animated=False):
    """
    Get the indexand f cost of the surrounding nodes of a given position.

    Parameters
    ----------
    node : ndarray
        The index and the g cost of the current node.
    start : ndarray
        The index of the start node.
    end : ndarray
        The index of the end node.
    grid : ndarray
        The grid where the path needs to be found.
    CLOSED : ndarray
        The list of the evaluated nodes.
        If a node has already been evaluated, then it will be skipped

    Returns
    -------
    ndarray
        Returns an array with all the new nodes
    """
    for row in [-1, 0, 1]:
        for col in [-1, 0, 1]:
            if not (row == col == 0):
                index = node[:2] + (row, col)

                if (index < 0).any():
                    continue

                elif index.tolist() in CLOSED[:, :2].tolist():
                    continue

                else:
                    try:
                        if grid[index[0], index[1]] == -1:
                            # Pass if neighbour is a wall
                            continue

                        g_cost = int(((index - start) ** 2).sum() ** 0.5 * 10)
                        h_cost = int(((index - end) ** 2).sum() ** 0.5 * 10)
                        f_cost = g_cost + h_cost * 2

                        if index.tolist() in OPEN[:, :2].tolist():
                            old_index = np.argwhere(OPEN[:, :2] == index)[0]
                            old_f_cost = OPEN[old_index[0], old_index[1]]
                            if f_cost >= old_f_cost:
                                continue
                            else:
                                OPEN = np.delete(OPEN, old_index, 0)

                        new_node = np.array([*index, f_cost, *node[:2]])
                        OPEN = np.append(OPEN, new_node)
                        OPEN = OPEN.reshape((OPEN.size // 5, 5))

                        if animated:
                            grid[index[0], index[1]] = 1

                    except IndexError:
                        # Pass if neighbour does not exist
                        continue

    return OPEN


def find_path(grid, inplace=False):
    """
    Uses A* Algorithm to find the shortest path between to points.

    Parameters
    ----------
    grid : ndarray
        The grid to look for the shortest path.
        There has to be a single starting point marked by a 1,
        a single end point, marked by a 2,
        and can contain walls, marked by a -1.
        The rest of the grid must be full of 0s
    inplace : bool, optional, defaults to False
        If True, will change the inputed grid directly
        and will not return anything.
        If it is False, it will not change the grid but created a copy of it
        and return the modified copy,
        it is useful if you want to conserve the initial grid.

    Returns
    -------
    ndarray
        The grid with the path included.
        The path is marked by 3s (including starting and ending node).
    """

    # Node = {row, col, h_cost, f_cost, parents_col, parents_row}

    if not inplace:
        grid = grid.copy()

    OPEN = np.array([], dtype=int)
    CLOSED = np.array([], dtype=int)

    start = np.argwhere(grid == 1)[0]
    end = np.argwhere(grid == 2)[0]

    OPEN = np.append(OPEN, start)
    OPEN = np.append(OPEN, 0)
    OPEN = np.append(OPEN, start).reshape(1, 5)

    exploring = True
    print('Exploring')
    i = 0
    while exploring:
        i += 1
        current_index = np.argmin(OPEN[:, 2])
        current = OPEN[current_index]

        OPEN = np.delete(OPEN, current_index, 0)
        CLOSED = np.append(CLOSED, current)
        CLOSED = CLOSED.reshape((CLOSED.size // 5, 5))

        OPEN = reveal_surrounding_nodes(
            current[:3], start, end, grid, OPEN, CLOSED)

        if end.tolist() in OPEN[:, :2].tolist():
            index = OPEN[:, :2].tolist().index(end.tolist())

            CLOSED = np.append(CLOSED, OPEN[index])
            CLOSED = CLOSED.reshape((CLOSED.size // 5, 5))

            exploring = False

    current_index = CLOSED[:, :2].tolist().index(end.tolist())

    retracing = True
    print('Retracing...')
    while retracing:
        current = CLOSED[current_index]
        grid[current[0], current[1]] = 3

        if current_index == 0:
            retracing = False

        current_index = CLOSED[:, : 2].tolist().index(current[3:].tolist())

    if not inplace:
        return grid


def find_path_animated(grid):
    """
    Uses A* Algorithm to find the shortest path between to points.

    Parameters
    ----------
    grid : ndarray
        The grid to look for the shortest path.
        There has to be a single starting point marked by a 1,
        a single end point, marked by a 2,
        and can contain walls, marked by a -1.
        The rest of the grid must be full of 0s
    """

    # Node = {row, col, f_cost, parents_col, parents_row}

    OPEN = np.array([], dtype=int)
    CLOSED = np.array([], dtype=int)

    start = np.argwhere(grid == 1)[0]
    end = np.argwhere(grid == 2)[0]

    OPEN = np.append(OPEN, start)
    OPEN = np.append(OPEN, 0)
    OPEN = np.append(OPEN, start).reshape(1, 5)

    exploring = True
    print('Exploring')
    i = 0
    while exploring:
        i += 1
        current_index = np.argmin(OPEN[:, 2])
        current = OPEN[current_index]

        OPEN = np.delete(OPEN, current_index, 0)
        CLOSED = np.append(CLOSED, current)
        CLOSED = CLOSED.reshape((CLOSED.size // 5, 5))

        grid[current[0], current[1]] = 2

        OPEN = reveal_surrounding_nodes(
            current[:3], start, end, grid, OPEN, CLOSED, True)

        if end.tolist() in OPEN[:, :2].tolist():
            index = OPEN[:, :2].tolist().index(end.tolist())

            CLOSED = np.append(CLOSED, OPEN[index])
            CLOSED = CLOSED.reshape((CLOSED.size // 5, 5))

            exploring = False

    current_index = CLOSED[:, :2].tolist().index(end.tolist())

    retracing = True
    print('Retracing...')
    while retracing:
        current = CLOSED[current_index]
        grid[current[0], current[1]] = 3

        if current_index == 0:
            retracing = False

        current_index = CLOSED[:, : 2].tolist().index(current[3:].tolist())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename',
                        default=None, type=str,
                        help='The name of the file containing the grid')
    args = parser.parse_args()

    if args.filename:
        with open(args.filename) as fp:
            initial_grid = json.load(fp)
            initial_grid = np.array(initial_grid)

        # Check if grid is in the correct format.
        invalid_values = initial_grid[(initial_grid < -1) |
                                      (initial_grid > 2)].astype(str)
        if invalid_values.tolist():
            raise ValueError(
                f'Grid contains invalid value(s) : {", ".join(invalid_values)}')

    else:

        initial_grid = np.zeros((9, 9))

        start = np.random.randint(0, 9, 2)
        initial_grid[start[0], start[1]] = 1

        end = np.random.randint(0, 9, 2)
        initial_grid[end[0], end[1]] = 2

    path = find_path(initial_grid)
    print(f'\nInitial grid:\n{initial_grid}')
    print(f'\nPATH: \n{path}')
