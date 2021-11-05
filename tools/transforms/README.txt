Here are tools that apply geometrical or photometrical transforms to 
numpy arrays. Photometrical transforms are functions that modify the scalar values at
the voxels but have no effect on the frame. On the other hand, geometrical transforms 
modify the voxel coordinates, while trying to retain the scalar values corresponding to
the voxel positions. Examples of photometrical transforms include statistical local 
filtering, whereas examples of geometrical transforms include rotations or translations.