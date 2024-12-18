# ORTools
Experiment with OR-Tools to solve transport problems. OR-Tools, an optimization library by Google, offers algorithms for combinatorial optimization. "OR" signifies "Operations Research," focusing on analytical methods for optimal decisions in complex scenarios.

1. The "execute" folder inside "problems" contain the 6 problems:
   CVRP (Capacitated Vehicle Routing Problem)
   VRPTW (Vehicle Routing Problem with Time Windows)
   VRPPD (Vehicle Routing Problem with Pickup and Delivery)
   DVRP (Distance Vehicle Routing Problem)
   MDVRP (Vehicle Routing Problem with Multiple Depots)
   TSP (Traveling Salesman Problem)


2. The "solutions" folder inside "problems" contain the solutions according to the instances evaluated
 in the problem, with distance Euclidean and Manhattan.
 Example: ("solutions_hfvrp" -> contains the results of the instances within 'hfvrp_instances')


3. The "variants" folder inside "problems" contains several examples of the problems
 such as time windows, break points, ...
 using the modules pywrapcp, routing_enums_pb2 of constraint_solver.


4. The "import_data" file inside "load_data" contains the code to import data from previous problem entries
 for each of the problems to be solved so that ORTools processes it with the structure
 it accepts. 


5. This data that is imported is found in the "instances" folder separated by folders
 according to the problems. 


6. The "run.py" file contains the code to run the program and call ORTools solvers