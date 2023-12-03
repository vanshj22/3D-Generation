import bpy

bpy.op.mesh.primitive_cube_add(size=4)

cube_object = bpy.context.active_object
cube_object.location.z = 5