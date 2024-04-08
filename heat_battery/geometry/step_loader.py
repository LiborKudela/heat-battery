from mpi4py import MPI
from math import pi
import gmsh
import os
from heat_battery.utilities import save_data
from inspect import getargspec
from textwrap import dedent

def build_geometry_from_stepfile(
        path,
        name='mesh',
        dir='meshes/experiment_inventor',   
        verbosity=0,
        groups={},
        mesh_size_max = 0.1,
        fltk=False,
        extract_axisymetry=True,
        probes_coords=[],
        probes_names=[],
        mats=[],
        bcs=[],
        step_scalling=0.001,
        custom_data={},
    ):
    if MPI.COMM_WORLD.rank == 0:

        file_path = dir + f'/{name}'
        gmsh_file = file_path + '.msh'
        add_data_file = file_path + '.ad'

        os.makedirs(dir, exist_ok=True)

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber('General.Verbosity', verbosity)

        gmsh.model.add("test_inventor")
        gmsh.logger.start()
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Geometry.OCCScaling", step_scalling)

        v = gmsh.model.occ.importShapes(path)
        # for ent in v:
            # print(gmsh.model.getEn(*ent))
        f_tags, f_dim_tags = gmsh.model.occ.fragment(v[0:1], v[1:])
        
        assert v == f_tags, dedent(f"""Non-matching fragments v={v}, f={f_tags}, 
                                   this often occures when using revolve in model definitions,
                                   try using extrusions instead!! """)

        if extract_axisymetry:
            # TODO: get dimensions of the plane from bounding box of the model
            xz_plane = [(2, gmsh.model.occ.addRectangle(0, -100, 0, 50, 200))]
            r = gmsh.model.occ.intersect(xz_plane, v)
            dim = 2
            jac_f = lambda x: 2*pi*x[0]
            for i, item in enumerate(probes_coords):
                probes_coords[i] = [item[0], item[2], 0.0]
        else:
            dim = 3
            jac_f = lambda x: 1

        gmsh.model.occ.synchronize()
        i = 1
        bc_names = []
        for bc_name, ents in bcs.items():
            gmsh.model.addPhysicalGroup(dim-1, ents, i, bc_name)
            i += 1
            bc_names.append(bc_name)

        # create selected p-groups
        i = 1
        for name, entities in groups.items():
            gmsh.model.addPhysicalGroup(dim, entities, i, name)
            i += 1

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size_max)
        gmsh.model.mesh.generate(dim)
        gmsh.write(gmsh_file)

        if fltk:
            gmsh.fltk.run()

        spec = getargspec(build_geometry_from_stepfile).args
        local_scope = locals()
        call_data = dict(zip(spec, [eval(arg, local_scope) for arg in spec]))

        add_data = {
            'call_data':call_data,
            'dim':dim,
            'probes_coords':probes_coords,
            'probes_names':probes_names,
            'materials':mats,
            'boundaries':bc_names,
            'jac_f':jac_f,
            }
        
        save_data(add_data_file, add_data)