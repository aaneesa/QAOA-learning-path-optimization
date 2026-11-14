# qaoa-learning-path-optimization

Comparing QAOA, greedy, fixed-curriculum, and exact solvers for personalized learning module selection under study-time and prerequisite constraints.

## Overview
A learner has limited study time and must choose which modules to study, where some modules require others first. We model this as a knapsack problem with prerequisite constraints, encode it as a QUBO/Ising Hamiltonian, and compare QAOA against a fixed curriculum, a greedy heuristic, and an exact solver.

All results come from classical simulation on synthetic instances and hold under the assumed benefit model. We do not claim quantum advantage.

## Research Questions
1. **Personalization:** Does optimizing module selection beat a fixed curriculum?
2. **Optimization:** How close does QAOA get to the optimum compared with greedy?
3. **Scalability:** How do modules, prerequisite density, and time tightness affect QAOA?