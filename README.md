# Multi-Agent Consensus with Delay – Python Simulation

This is a Python script that simulates a group of agents trying to reach consensus even when there's a communication delay between them. The agents move in 3D space and follow a double integrator dynamic model.

## What It Does

- Simulates `N` agents connected in a ring graph
- Each agent estimates its own state using an observer (Luenberger type)
- Communication between agents is delayed by `delta` time steps
- Despite the delay, all agents manage to agree (reach consensus)
- Plots:
  - Position and velocity over time
  - Estimation error and control input
  - 3D trajectories with initial network connections
