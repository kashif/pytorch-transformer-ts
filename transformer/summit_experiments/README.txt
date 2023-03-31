
#### Documentation of summit: <br>https://docs.olcf.ornl.gov/systems/summit_user_guide.html.
To setup password and initial steps:
https://docs.olcf.ornl.gov/connecting/index.html#connecting-for-the-first-time
Also, set up jupyter notebook: <br>https://docs.olcf.ornl.gov/services_and_applications/jupyter/overview.html#jupyter-at-olcf.

#### 
Load module:
module load cuda
module load job-step-viewer
module load ums
module load ums-gen119
module load nvidia-rapids/cucim_21.08

To make conda environment with requirement.txt
Make changes in script.bash -> change output filepath and environment name in line 15.
