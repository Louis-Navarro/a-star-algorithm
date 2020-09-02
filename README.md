A* algorithm
============

This repository is a A* algorithm I made.

How to download
---------------

1. Clone this repository using `git clone https://github.com/Louis-Navarro/a-star-algorithm.git` or download the `.zip` file. Then open a terminal window in the folder

2. Type `pip install -r requirements.txt` and then press enter. This will install all the required modules

3. Done ! You can now visualize an A* algorithm with a GUI and easily find paths in mazes !

How to use
----------

* `main.py` is a program that opens a GUI (using `pygame`) where you can create a maze.
  * To add a wall, click the middle mouse button (be careful, the algorithm can cross two diagonal walls).
  * To place the starting point, click the left mouse button
  * To place the ending point, click the right mouse button
  * To run the algorithm, press the enter key on your keyboard.
  * To clean the grid (except the start and end points), press the escape key on your keyboard.
* `algorithm.py` is a program that contains the A* algorithm, where you can find a path
  * The program will created a grid with :
    * A starting point, randomly generated.
    * An ending point, also randomly generated.
  * It will then print the grid, run the algorithm and print th egrid with the path
