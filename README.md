# HurricaneLossModel


### Process

Historical Data → Frequency Model → Severity Model → Loss Function

Severity Model → Monte Carlo Simulation → Annual Loss Distribution → Reinsurance Layer Analysis


 ### Data set
 
 Downloaded from: https://www.aoml.noaa.gov/hrd/hurdat/Data_Storm.html


### Pseudo Code

1.	Define the inputs
   a.	Hurricane model: Define the mean and variance of the number of hurricanes for the simulated year
   b.	Wind Model: Use a log-normal distribution model to define the shape, loc, and scale of the wind model curve
   c.	Exposure: Define the exposure to be used in the loss model
2.	Sample the number of hurricanes during the simulated year using the hurricane model with a negative binomial distribution
3.	For each hurricane in step 2,
   a.	Sample the wind speed at landfall using the wind model and a log normal distribution
   b.	Compute the damage ratio
   c.	Compute the event loss
   d.	Aggregate the loss of all hurricanes
4.	Repeat steps 2-3 for N number of times (baseline N = 100,000)
