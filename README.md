# ORTools
Experiment with OR-Tools to solve transport problems. OR-Tools, an optimization library by Google, offers algorithms for combinatorial optimization. "OR" signifies "Operations Research," focusing on analytical methods for optimal decisions in complex scenarios.

1. The vrp, cvrp and tsp folders contain the 3 problems:
   VRP (Vehicle Routing Problem)
   CVRP (Capacitated Vehicle Routing Problem)
   TSP (Traveling Salesman Problem)


2. The examples folder within these folders contains several examples of the problems
 such as time windows, pickup and delivery, ...
 using the modules pywrapcp, routing_enums_pb2 of constraint_solver.


3. The "import_data" file contains the code to import data from previous problem entries
 for each of the problems to be solved so that ORTools processes it with the structure
 it accepts.


4. This data that is imported is found in the "instances" folder separated by folders
 according to the problems.


5. The "variants" folder within "examples" contains the algorithms to solve the variants
 of the problem in question.


6. The "solutions" folders contain the solutions according to the instances evaluated
 in the problem. 
 Example: ("solutions_bhcvrp" -> contains the results of the instances within "bhcvrp_instances")