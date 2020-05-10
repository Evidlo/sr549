# ECE549 Super-resolution Project

## Install

    pip install -e .
    
## Directory Structure

- `sr549/` - reusable functions and algorithm implementations go here.  can be imported from the `sr549` module after installing with pip
- `code/` - numerical experiments, figure generation

## ipython-cells

    >>> !pip install ipython-cells
    >>> %load ipython_cells
    >>> %load_file example.py

    # run the "register" cell
    >>> %cell_run register
    
    # run all cells after "register" inclusive
    >>> %cell_run register$

    # run all cells before "register" inclusive
    >>> %cell_run ^register
