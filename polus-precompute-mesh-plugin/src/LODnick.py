import numpy as np
from pathlib import Path
from skimage import measure

import argparse, logging, subprocess, time, multiprocessing
from bfio import BioReader, BioWriter, JARS, LOG4J
import bioformats
import javabridge as jutil

import struct,json
from struct import *
import traceback
import trimesh
import math
import pandas
import shutil

# Conversions
# Conversion factors to nm, these are based off of supported Bioformats length units
UNITS = {'m':  10**9,
         'cm': 10**7,
         'mm': 10**6,
         'µm': 10**3,
         'nm': 1,
         'Å':  10**-1}

# Needs to be specified
input_path = Path('/home/schaubnj/Desktop/Projects/polus-plugins/input_volume/')
dataname = input_path.joinpath('dA30_5_dA30.ome.tif')
output_path = Path('/home/schaubnj/Desktop/Projects/polus-plugins/neuroglancer/dist/dev/dA30_5_dA30.Labels.ome.tif/dA30_5_dA30.Mesh.ome.tif')

if output_path.exists():
    shutil.rmtree(str(output_path))
output_path.mkdir(exist_ok=True)

# Bit depth of the draco coordinates, must be 10 or 16
bit_depth = 10

# Create two levels of detail
num_lods = 2

# Merge verticies that are closer than 1 pixel
trimesh.constants.tol.merge = 1

# Start the JVM
# jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(LOG4J))],class_path=JARS)
fragoffsum = 0

#checking to see if I login
try:
    # Load the image
    br = BioReader(dataname,backend='python')
    
    # Create the info file
    x_size, x_units = br.ps_x
    y_size, y_units = br.ps_y
    z_size, z_units = br.ps_z
    
    x_size *= UNITS[x_units]
    y_size *= UNITS[y_units]
    if None in [z_size,z_units]:
        z_size = (x_size + y_size) // 2
        z_units = x_units
    z_size *= UNITS[z_units]
    
    with open(str(output_path.joinpath("info")), 'w') as info:
        jsoninfo = {
            "@type" : "neuroglancer_multilod_draco",
            "lod_scale_multiplier" : 1,
            "vertex_quantization_bits" : bit_depth
        }
        info.write((json.dumps(jsoninfo)))
    volume = br[:].squeeze()

    # Get the ids for each segment
    IDS = np.unique(volume)
    
    # Master draco file offset
    fragment_offset = 0

    # need to create a for loop for all the ids.
    for iden in IDS[1:]:
        
        fragment_offsets = []
        fragment_positions = []
        num_fragments_per_lod = []
        vertex_offsets = []
        lod_scales = []
        
        chunk_shape = None
        
        for i in range(num_lods):
            fragcount = 0
            concatmesh = 0
            fragment_positions.append([])
            fragment_offsets.append([])
            lod_scales.append(x_size * float(2 ** (i)))
            
            num_fragments_per_lod.append(0)
            vertex_offsets.append([0.5*(2 ** i) for _ in range(3)])
        
            # Create the mesh
            vertices,faces,_,_ = measure.marching_cubes_lewiner((volume==IDS[iden]).transpose(1,0,2), step_size=(num_lods - i)*2+1)

            # getting the dimensions of the segment
            min_bounds = vertices.min(axis=0)
            max_bounds = vertices.max(axis=0)
            dim = max_bounds - min_bounds
            root_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Why does this work?? 
            # I thought step size was user defined??
            zstep = dim[2]/(2 ** (num_lods - i - 1))
            ystep = dim[1]/(2 ** (num_lods - i - 1))
            xstep = dim[0]/(2 ** (num_lods - i - 1))
            
            if isinstance(chunk_shape,type(None)):
                chunk_shape = (x_size * np.asarray([xstep,ystep,zstep])).astype(np.float32)
                # chunk_shape = np.asarray([xstep,ystep,zstep]).astype(np.float32)

            for z in np.arange(min_bounds[2],max_bounds[2],zstep):
            
                z_section = trimesh.intersections.slice_mesh_plane(root_mesh,
                                                                (0.0,0.0,1.0),
                                                                (0.0,0.0,float(z)))
                
                z_section = trimesh.intersections.slice_mesh_plane(z_section,
                                                                (0.0,0.0,-1.0),
                                                                (0.0,0.0,float(z + zstep)))
    
                for y in np.arange(min_bounds[1],max_bounds[1],ystep):
                    
                    y_section = trimesh.intersections.slice_mesh_plane(z_section,
                                                                        (0.0,1.0,0.0),
                                                                        (0.0,float(y),0.0))
                    
                    y_section = trimesh.intersections.slice_mesh_plane(y_section,
                                                                        (0.0,-1.0,0.0),
                                                                        (0.0,float(y + ystep),0.0))

                    for x in np.arange(min_bounds[0],max_bounds[0],xstep):
                    
                        x_section = trimesh.intersections.slice_mesh_plane(y_section,
                                                                            (1.0,0.0,0.0),
                                                                            (float(x),0.0,0.0))
                                                    
                        x_section = trimesh.intersections.slice_mesh_plane(x_section,
                                                                            (-1.0,0.0,0.0),
                                                                            (float(x + xstep),0.0,0.0))
                        
                        # get the local chunk shape
                        if len(z_section.vertices) == 0:
                            continue
                    
                        fragment_positions[-1].append([       
                            (x-min_bounds[0]) / xstep,
                            (y-min_bounds[1]) / ystep,
                            (z-min_bounds[2]) / zstep,
                        ])
                        
                        # Transformation matrix to translate and scale
                        scale =  np.asarray([xstep,ystep,zstep,1]) / (2 ** bit_depth - 1)
                        transform = np.asarray([[1, 0, 0, -x/xstep],
                                                [0, 1, 0, -y/ystep],
                                                [0, 0, 1, -z/zstep],
                                                [0, 0, 0,  0]]) / scale
                        z_section.apply_transform(transform)

                        drcfile = output_path.joinpath(str(iden))
                        with open(str(drcfile), "ab+") as draco:
                            start = draco.tell()
                            draco.write(trimesh.exchange.ply.export_draco(mesh=z_section, bits=bit_depth)) # bit must match vertex_quantization_bits
                            num_fragments_per_lod[-1] += 1
                            fragment_offsets[-1].append(draco.tell() - start)
                            fragcount = fragcount + 1

        num_fragments_per_lod = np.asarray(num_fragments_per_lod).astype('<I')
        gridorigin = x_size * min_bounds # when fragment position is all zeros, this needs to change
        # gridorigin = min_bounds # when fragment position is all zeros, this needs to change
        manifest_file = output_path.joinpath((str(iden)+".index"))
        vertex_offsets = np.asarray(vertex_offsets).astype('<f')
        with open(str(manifest_file), 'wb') as index:
            index.write(chunk_shape.astype('<f').tobytes(order='C'))
            index.write(gridorigin.astype('<f').tobytes(order="C"))
            index.write(struct.pack("<I",num_lods))
            index.write(np.asarray(lod_scales).astype('<f').tobytes(order="C"))
            index.write(vertex_offsets.tobytes(order="C"))
            index.write(num_fragments_per_lod.astype('<I').tobytes(order="C"))
            for i in range(0, num_lods):
                fp = np.asarray(fragment_positions[i]).astype('<I')
                index.write(fp.astype('<I').tobytes(order="C"))
                fo = np.asarray(fragment_offsets[i]).astype('<I')
                index.write(fo.tobytes(order="C"))

except Exception as e:
    traceback.print_exc()
finally:
    jutil.kill_vm()
