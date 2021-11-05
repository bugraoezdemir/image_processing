Here are tools that apply geometrical or photometrical transforms to 
numpy arrays. Photometrical transforms modify the voxel values 'in-place', i.e., they
do not change the voxel positions. 

On the other hand, geometrical transforms modify the voxel positions, while trying to
retain the voxel values. 

Examples of photometrical transforms include filters such as Gaussian and Laplacian 
filters, whereas examples of geometrical transforms include rotations, projections, etc.