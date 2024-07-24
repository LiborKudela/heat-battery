## Example_01 but from step file

This example is identical to Example_01, but the mesh is generated from .step
file instead for using gmsh directly. The step file has been created using 
CAD software (Autodesk Inventor Profesional 2024). 

This is more convenient when the geometry becomes complicated with a lot of
features and it is not necessarry to generate it algorithmically for parameter
sweeps.

## Axisymetry from 3-dimensional model automaticaly

The axisymetry can be extracted by keyword argument `extract_axisymetry=True`,
passed to the function `build_geometry_from_stepfile`
(see [geometry.py](geometry.py)). The object (3d model) in the .step file must
be placed such that the axis of axysymetry is the same as the Z-axis.

The subdomains are extracted by calculation an intersection of the model with a 
vertical plane. In order to corectly select entities (for `mats` and `bcs`) you
can use keyword argument `fltk=['premesh']`, which opens a GMSH window with
the model loaded and after the plane intersection was caclulated beu before the
mesh is generated, which allows easier visual orientation. By hovering a mouse
over the entities, their integer numbering can be identified.

The 'view' tab can be also use to find the entities by selectively 
hiding entities
(for more see [gmsh docs](https://gmsh.info/dev/doc/texinfo/gmsh.pdf)).
