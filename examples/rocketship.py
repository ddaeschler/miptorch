import torch
from miptorch import mip_solve

# mip program to optimize the landing of a rocket

dtype = torch.float32

timesteps: int = 10

v0 = 100
distace_to_target0 = 1000
starting_fuel = 500
G = 9.8

# start first modeling fuel. our remaining fuel is the fuel at the end of the
# previous timestep i - 1 minus our thrust in this timestep
# fuel[0] = starting_fuel - thrust[0]
# fuel[i] = fuel[i - 1] - thrust[i]
# fuel[i] >= 0

# so we model as (having <= only)
# fuel[0] + thrust[0] <= starting_fuel
# -fuel[0] - thrust[0] <= -starting_fuel
# fuel[i] - fuel[i - 1] + thrust[i] <= 0
# -fuel[i] + fuel[i - 1] - thrust[i] <= 0
# -fuel[i] <= 0

constraint_rhs = [starting_fuel, -starting_fuel]
rows = [] 

def zeros_row(timesteps, cols):
    return [0] * timesteps * cols

def fuel_col(curr_time):
    return curr_time

def thrust_col(curr_time):
    return timesteps + curr_time

for i in range(timesteps):
    if i == 0:
        # 2 cols per timestep: fuel and thrust
        # laid out as f0,f1,f2,...,fn,t0,t1,t2,...,tn
        next_row = zeros_row(timesteps, 2) # 2 cols per timestep: fuel and thrust
        next_row[fuel_col(0)] = 1.0
        next_row[thrust_col(0)] = 1.0
        rows.append(next_row)

        next_row = zeros_row(timesteps, 2)
        next_row[fuel_col(0)] = -1.0
        next_row[thrust_col(0)] = -1.0
        rows.append(next_row)

    else:
        next_row = zeros_row(timesteps, 2)
        next_row[fuel_col(i)] = 1.0
        next_row[fuel_col(i - 1)] = -1.0
        next_row[thrust_col(i)] = 1.0
        rows.append(next_row)
        constraint_rhs.append(0.0)

        next_row = zeros_row(timesteps, 2)
        next_row[fuel_col(i)] = -1.0
        next_row[fuel_col(i - 1)] = 1.0
        next_row[thrust_col(i)] = -1.0
        rows.append(next_row)
        constraint_rhs.append(0.0)

        next_row = zeros_row(timesteps, 2)
        next_row[i] = -1.0
        rows.append(next_row)
        constraint_rhs.append(0.0)


# next we model the fuel and velocity interaction
# our velocity is equal to our velocity at the previous timestep plus gravity now
# minus any thrust that was applied
# velocity[i] = velocity[i - 1] - thrust[i]
# velocity[i] - velocity[i - 1] + thrust[i] <= 0

# we'll need to append columns to the rows already started
for row in rows:
    row.extend([0.0] * timesteps)

def veocity_column(curr_time):
    return (2*timesteps) + curr_time

for i in range(timesteps):
    pass
