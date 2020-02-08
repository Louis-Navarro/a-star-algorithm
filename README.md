A* algorithm
============
This repository is a A* algorithm I made.


## How to download
1. Clone this repository using `git clone https://github.com/Louis-Navarro/a-star-algorithm.git` or download the `.zip` file. Then open a terminal window in the folder

2. Type pip install -r requirements.txt and ther press enter. This will install all the required modules

3. Done ! You can now visualize an A* algorithm with a GUI and easily find paths in mazes !

## How to use
* `main.py` is a program that opens a GUI (using `pygame`) where you can create a maze.
  * To add a wall, click the left mouse button (be careful, the algorithm can cross two diagonal walls).
  * When you click the left mouse button, you will place
    * The startng point the first time.
    * The ending point the second time.
  * To run the algorithm, press the enter key on your keyboard.
  * To clean the grid, press the escape key on your keyboard.
* `algorithm.py` is a program that contains the A* algorithm, where you can find a path
  * When you launch the program, a grid will be created with :
    * A starting point, randomly generated.
    * An ending point, also randomly generated.
  * After creating the grid, the algorithm will find the best path and return it
  * After it finds the best path, it will display :
    * The initial grid.
    * The path the algorithm found.
