# cohda
The Combinatorial Optimization Heuristic for Distributed Agents (COHDA) was developed as part of my [PhD thesis](http://oops.uni-oldenburg.de/1960/) (in German).

*Abstract:* In distributed combinatorial optimization problems, the underlying decision variables are usually constrained by interdependencies. For a successful optimization, a coordination strategy based on information exchange is thus necessary. An example for such a problem is the day-ahead planning of controllable distributed energy units (DEU). Here, each DEU has to select its own schedule for the planning horizon in such a way, that the schedules of all units jointly match a given global target profile as close as possible. For this purpose, the heuristic COHDA (Combinatorial Optimization Heuristic for Distributed Agents) is developed, which realizes a completely distributed combinatorial optimization process in an asynchronous communication environment. The correctnes of COHDA with respect to convergence and termination is proven formally. Simulation experiments show the robustness and scalability of the approach, as well as its effectiveness in a number of smart grid planning scenarios.