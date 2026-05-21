AIPP simulation running scripts + Diffusion model scripts for training and sampling. 
Diffusion model can be found in ./Diffusion/SparseDiffusion.py. Training params are all in the head of the file.

SparseDiffusion.py has inputs: 
Mean conditions (1, 51, 51)
Variance conditions (1, 51, 51)
Initial position (1, 2)

And outputs the expected noise in a control waypoint vector (1, 2, 8), which is used to find a CubicSpline. 

Loss is computed by MSE loss between expected waypoints, as well as noisy splines. The principle behind the spline loss is that any resultant spline is the sum of a noisy spline + the true spline that we're trying to converge to.
