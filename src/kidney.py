# Generate kidney tubule geometry using Python in Blender

import bpy
import numpy as np
import mathutils
import os

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

def create_tubule(start_point, direction, length, diameter, turns=3):
    """
    Create a single winding tubule (proximal convoluted tubule)
    """
    curve = bpy.data.curves.new('tubule_curve', 'CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = diameter / 2  # Tubule radius
    curve.resolution_u = 12
    
    spline = curve.splines.new('NURBS')
    
    # Generate winding path
    num_points = 20
    points = []
    
    for i in range(num_points):
        t = i / num_points
        
        # Sinusoidal winding path
        x = start_point[0] + direction[0] * length * t
        y = start_point[1] + direction[1] * length * t + np.sin(t * turns * 2 * np.pi) * 0.2
        z = start_point[2] + direction[2] * length * t + np.cos(t * turns * 2 * np.pi) * 0.2
        
        points.append((x, y, z))
    
    spline.points.add(len(points) - 1)
    for i, point in enumerate(points):
        spline.points[i].co = (point[0], point[1], point[2], 1)
    
    # Create object from curve
    obj = bpy.data.objects.new('tubule', curve)
    bpy.context.collection.objects.link(obj)
    
    return obj

def create_glomerulus(center, radius):
    """
    Create a spherical glomerulus
    """
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        location=center,
        segments=32,
        ring_count=16
    )
    glom = bpy.context.active_object
    glom.name = 'glomerulus'
    
    # Add texture (capillary network appearance) - OCT colors
    mat = bpy.data.materials.new(name="Glomerulus_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    # Darker reddish-brown for glomeruli in OCT
    bsdf.inputs['Base Color'].default_value = (0.6, 0.3, 0.2, 1)
    bsdf.inputs['Roughness'].default_value = 0.8

    glom.data.materials.append(mat)
    
    return glom

def create_bowmans_capsule(center, inner_radius, outer_radius):
    """
    Create Bowman's capsule (double sphere)
    """
    # Outer capsule
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=outer_radius,
        location=center,
        segments=32,
        ring_count=16
    )
    outer = bpy.context.active_object
    outer.name = 'bowmans_capsule_outer'
    
    # Inner cavity (subtract)
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=inner_radius,
        location=center,
        segments=32,
        ring_count=16
    )
    inner = bpy.context.active_object
    inner.name = 'bowmans_capsule_inner'
    
    # Use boolean modifier to create hollow capsule
    mod = outer.modifiers.new(name="Boolean", type='BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.object = inner
    
    bpy.context.view_layer.objects.active = outer
    bpy.ops.object.modifier_apply(modifier="Boolean")
    bpy.data.objects.remove(inner, do_unlink=True)
    
    # Material - OCT appearance
    mat = bpy.data.materials.new(name="Capsule_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = (0.8, 0.6, 0.3, 1)  # Golden
    bsdf.inputs['Alpha'].default_value = 0.5  # Semi-transparent
    mat.blend_method = 'BLEND'

    outer.data.materials.append(mat)
    
    return outer

# BUILD THE KIDNEY CORTEX MODEL
# ================================

# Scale: 100× magnification (easier to print)
# 1 mm tissue = 100 mm model = 10 cm

base_scale = 100  # 100× scale-up
tubule_diameter = 0.05 * base_scale  # 50 μm → 5 mm (realistic proximal tubule)
glomerulus_radius = 0.1 * base_scale   # 100 μm → 10 mm
tissue_size = 1.5 * base_scale         # 1.5 mm → 150 mm (15 cm) - smaller for density

# Create base tissue block (for context)
bpy.ops.mesh.primitive_cube_add(
    size=tissue_size,
    location=(tissue_size/2, tissue_size/2, tissue_size/2)
)
base = bpy.context.active_object
base.name = 'tissue_base'

# Make it very transparent (just for reference)
mat = bpy.data.materials.new(name="Tissue_Material")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs['Base Color'].default_value = (0.2, 0.2, 0.2, 1)
bsdf.inputs['Alpha'].default_value = 0.05  # Almost invisible
mat.blend_method = 'BLEND'
base.data.materials.append(mat)

# Create tubule network (proximal convoluted tubules)
np.random.seed(42)

num_tubules = 150  # Dense tissue like OCT image
tubules = []

# Create grid-based distribution for realistic density
grid_size = int(np.sqrt(num_tubules))
spacing = (tissue_size - 40) / grid_size

for i in range(num_tubules):
    # Grid position with slight random offset for natural appearance
    grid_x = (i % grid_size) * spacing + 20
    grid_y = (i // grid_size) * spacing + 20

    # Add randomness to avoid perfect grid
    start_x = grid_x + np.random.uniform(-spacing*0.3, spacing*0.3)
    start_y = grid_y + np.random.uniform(-spacing*0.3, spacing*0.3)
    start_z = tissue_size - 10  # Near top

    start_point = (start_x, start_y, start_z)

    # Direction (mostly straight down with minimal winding - like real tubules)
    direction = (
        np.random.uniform(-0.1, 0.1),  # Less lateral movement
        np.random.uniform(-0.1, 0.1),
        -1.0
    )

    # Normalize direction
    dir_length = np.sqrt(sum(d**2 for d in direction))
    direction = tuple(d / dir_length for d in direction)

    # Create tubule (longer, less winding)
    tubule_length = np.random.uniform(tissue_size * 0.8, tissue_size * 1.2)
    tubule = create_tubule(
        start_point,
        direction,
        tubule_length,
        tubule_diameter,
        turns=np.random.randint(1, 3)  # Less winding
    )

    # Material (OCT-like colors: yellow/gold for viable, darker for ischemic)
    mat = bpy.data.materials.new(name=f"Tubule_{i}_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]

    # OCT imaging colors: gold/yellow tones
    viability = np.random.random()
    if viability > 0.3:  # 70% viable
        # Yellow-gold viable tissue
        bsdf.inputs['Base Color'].default_value = (
            np.random.uniform(0.8, 1.0),   # R
            np.random.uniform(0.6, 0.8),   # G
            np.random.uniform(0.1, 0.3),   # B
            1
        )
    else:  # 30% damaged/ischemic
        # Darker reddish-brown damaged tissue
        bsdf.inputs['Base Color'].default_value = (
            np.random.uniform(0.5, 0.7),   # R
            np.random.uniform(0.2, 0.4),   # G
            np.random.uniform(0.1, 0.2),   # B
            1
        )

    bsdf.inputs['Roughness'].default_value = 0.7
    tubule.data.materials.append(mat)
    tubules.append(tubule)

# Create glomeruli (renal corpuscles) - fewer for cortex cross-section
num_glomeruli = 8  # ~5% of tubules have associated glomeruli visible

for i in range(num_glomeruli):
    # Random position (avoid edges)
    glom_x = np.random.uniform(30, tissue_size - 30)
    glom_y = np.random.uniform(30, tissue_size - 30)
    glom_z = np.random.uniform(tissue_size * 0.3, tissue_size * 0.7)

    center = (glom_x, glom_y, glom_z)

    # Create renal corpuscle
    capsule = create_bowmans_capsule(
        center,
        inner_radius=glomerulus_radius * 1.15,
        outer_radius=glomerulus_radius * 1.35
    )

    glomerulus = create_glomerulus(center, glomerulus_radius)

print(f"Created dense kidney cortex model (OCT-style) with {num_tubules} tubules and {num_glomeruli} glomeruli")
print(f"Model size: {tissue_size:.1f} mm³")
print(f"Tubule density: ~{num_tubules / (tissue_size/100)**2:.0f} tubules/mm²")
print(f"Ready for export to STL")

# Export to STL (commented out - enable STL addon first)
# output_path = os.path.join(os.path.expanduser("~"), "kidney_oct_model.stl")
# bpy.ops.export_mesh.stl(filepath=output_path, use_selection=False)
# print(f"Exported to: {output_path}")
print("Model created successfully! Use File > Export > STL to save manually.")
