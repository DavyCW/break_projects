# break_projects
Repository for projects done while on break.

# How to Use:
All the classes with visualizations use their __call__ method to generate those
visualizations. If generating within a Jupyter Notebook, pass __call__
'notebook'. If generating from the command line in a container, pass __call__
'container'.

# Generate Data
src/classifiy3D/generate_data.py takes a ssp{n}.txt file from a folder called
data/ with the format:

"""radius (float)

1(int)   x_1(float)   y_1(float)   z_1(float)

...

i(int)   x_i(float)   y_i(float)   z_i(float)

...

n(int)   x_n(float)   y_n(float)   z_n(float)

"""

where n is the number of points. An example file is ssp256.txt, which has
256 points. These points represent the centers of spheres inside the unit
sphere, all with the same radius, also given in the file. They are the packing
of n spheres with maximal radius inside the unit sphere.

GenerateData then randomly generates a number of uniformly distributed points
inside these spheres, generated through a random theta, a random phi, and a
random radius, solving the three dimensional problem by taking the cube root of
the random generation before multiplying it be the sphere's radius.

Calling GenerateData plots the spheres and the randomly generated points inside
of them.