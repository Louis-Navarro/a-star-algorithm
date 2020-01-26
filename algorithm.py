import numpy as np


def get_smallest_node(node, CLOSED):
    """
    Get the node with the lowest f cost arround the given node.

    Parameters
    ----------
    node : ndarray
        The index of the current node.
        The nodes looked for will be surrounding this node
    CLOSED : ndarray
        The list of evaluated nodes.
        It will look for every node arround the given one
        AND in the CLOSED list.

    Returns
    -------
    ndarray
        The index of the node with the lowest f cost
    """
    # Set smallest to the greatest number + one
    arround = np.array([])

    CLOSED_LIST = CLOSED.tolist()
    INDEX_LIST = CLOSED[:, :2].tolist()

    for x in [0, 1, -1]:
        for y in [0, 1, -1]:
            if not (x == y == 0):
                # If x == y == 0, then we are evaluating the current node
                index = (node + (x, y)).tolist()
                if index in INDEX_LIST:
                    # If the neighbour was evaluated, add it to the list
                    # Add the index of the number
                    arround = np.append(arround, index)
                    # Get the f cost
                    f_cost_index = INDEX_LIST.index(index)
                    f_cost = CLOSED_LIST[f_cost_index][-1]
                    # Add the f cost
                    arround = np.append(arround, f_cost)

    arround = arround.reshape((arround.size // 3, 3))
    return arround[arround[:, -1].argmin()]


def reveal_surrounding_nodes(node, start, end, grid, CLOSED, inplace=True):
    """
    Get the indexand f cost of the surrounding nodes of a given position.

    Parameters
    ----------
    node : ndarray
        The index of the node.
    start : ndarray
        The index of the starting node, will be used to get the g cost.
    end : ndarray
        The index of the starting node, will be used to get the h cost.
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

    nodes = np.array([])

    for x in [0, 1, -1]:
        for y in [0, 1, -1]:
            if not (x == y == 0):
                # If x == y == 0, then we are evaluating the current node
                index = (node + (x, y)).astype(int)
                if index.tolist() in CLOSED[:, :2].tolist():
                    # Pass if neighbour has already been evaluated
                    continue

                else:
                    try:
                        new_node = grid[index[0], index[1]]
                        if new_node == -1:
                            # Pass if neighbour is a wall
                            continue

                        else:
                            g_cost = ((start - index) ** 2).sum() ** 0.5 * 10
                            h_cost = ((end - index) ** 2).sum() ** 0.5 * 10
                            f_cost = g_cost + h_cost

                            new_node = np.array([*index, f_cost])
                            nodes = np.append(nodes, new_node)

                    except IndexError:
                        # If the node does not exist
                        continue

    nodes = nodes.reshape((nodes.size // 3, 3))
    return nodes


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
        The path is marked by 3s (exculuding starting and maending node).
    """

    # Node = {col, row, f_cost}

    if not inplace:
        grid = grid.copy()

    OPEN = np.array([])
    CLOSED = np.array([])

    start = np.argwhere(grid == 1)[0]
    OPEN = np.append(OPEN, start)
    OPEN = np.append(OPEN, 0).reshape(1, 3)

    end = np.argwhere(grid == 2)[0]

    exploring = True
    print('Exploring...')
    while exploring:
        current_index = np.argmin(OPEN[:, -1])
        current = OPEN[current_index]

        OPEN = np.delete(OPEN, current_index, 0)
        CLOSED = np.append(CLOSED, current)
        CLOSED = CLOSED.reshape((CLOSED.size // 3, 3))

        if (current[:2] == end).all():
            exploring = False
            continue

        nodes = reveal_surrounding_nodes(
            current[:2], start, end, grid, CLOSED)
        OPEN = np.append(OPEN, nodes)
        OPEN = OPEN.reshape((OPEN.size // 3, 3))

    current = np.argwhere(grid == 2)[0]
    path = np.array([current])

    index = CLOSED[:, :2].tolist().index(current.tolist())
    CLOSED = np.delete(CLOSED, index, 0)

    retrace = True
    print('Retracing...')
    while retrace:
        current = get_smallest_node(current, CLOSED)[:2]
        path = np.append(path, current)

        index = CLOSED[:, :2].tolist().index(current.tolist())
        CLOSED = np.delete(CLOSED, index, 0)

        if (current == start).all():
            retrace = False

    path = path.reshape((path.size // 2, 2)).astype(int)
    grid[path[:, 0], path[:, 1]] = 3

    if not inplace:
        return grid


if __name__ == "__main__":
    initial_grid = np.zeros((9, 9))

    start = np.random.randint(0, 9, 2)
    initial_grid[start] = 1

    end = np.random.randint(0, 9, 2)
    initial_grid[end] = 2

    path = find_path(initial_grid)
    print(f'\nInitial grid:\n{initial_grid}')
    print(f'\nPath:\n{path}')
