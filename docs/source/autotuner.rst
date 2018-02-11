Autotuner
=========

The genetic-algorithm-based autotuner tries to optimize a TC by tuning the available mapping options. 

Each autotuning session starts with a set (population) of candidate options
which can be initialized randomly and/or from known starting points. Each
candidate is benchmarked and the best ones have a higher chance of surviving
and breeding to produce the next generation of candidates. This procedure is
repeated for a predefined number of generations. In the end the best candidate
is returned.

At the end of each generation new candidates must be selected. Each candidate
is either a combination of parent candidates (crossover) or one that survives
from the previous generation. Both types are potentially randomly changed
(mutation).  The top candidates (elites) survive intact (without mutations)
between generations. 

Parameters for Autotuning
-------------------------

The parameters that control the autotuner's behavior are the following:

* **Number of generations**: The number of tuning generation to be run.
* **Population size**: The number of candidates in each generation.
* **Number of elites**: The number of best candidates that are preserved intact between generations (without any mutations).
* **Crossover rate**: The rate at which new candidates are bred
  instead of just surviving across generations.
* **Mutation rate**: The rate at which candidate options are randomly
  changed (mutated).
* **Number of threads**: The number of threads that are used to compile different
  candidates in parallel.
* **GPUs**: A comma separated list of GPUs (ids) to use for evaluating
  candidates (e.g., "0,1,2,3").
* **RNG state**: The state used to seed the tuner's RNG.
* **Proto**: A protobuf filename to (re)store compilation results and
  profiling information of the candidate solutions.
* **min_launch_total_threads**: ???related to pruning???

Caching
------------

After each autotuning session the best candidates' profiling information and compilation results are stored in a cache. They can be subsequently retrieved to seed a new autotuning session.

Early Pruning
-------------
