# Ouroboros: Virtualized Queues for dynamic memory management
This repository holds the source code for Ouroboros, a dynamic, GPU memory manager based on fast queues.
For more details, you can watch our talk [here](https://www.youtube.com/watch?v=AoodGDFaiG4) or read the paper [here](https://dl.acm.org/doi/10.1145/3392717.3392742)

# Setup
* `mkdir build && cd build`
* `cmake ..` -> pass in CC or call `ccmake` for visual selection
* `make`
This generates 6 different executables, one for each possible instantiation of Ouroboros.


# ouroGraph
Can be found at [here](https://github.com/GPUPeople/ouroGraph)