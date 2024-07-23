from problems.problem_type import ProblemType, execute_problem

############
# RUN
############

for i in range(3, 10):
    execute_problem(ProblemType.VRPPD, i)
