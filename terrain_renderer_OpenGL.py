import sys
import os
import numpy as np
from PIL import Image
import math

try:
    from OpenGL.GL import *
    from OpenGL.GLUT import *
    from OpenGL.GLU import * # We'll use gluPerspective, but matrix math is better modern practice
except ImportError:
    print("ERROR: PyOpenGL or GLUT not found. Please install PyOpenGL, PyOpenGL-accelerate, and FreeGLUT.")
    sys.exit(1)



# --- Configuration ---
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
VERTICAL_EXAGGERATION = 5.0
DOWNSAMPLE_FACTOR = 4      # Increase for performance on large images (e.g., 2, 4, 8)
ROTATION_SPEED = 0.2
USE_LIGHTING = False # Start with lighting disabled for testing
INVERT_HEIGHTMAP = False # Set to True if black=high, white=low in your map
COLORMAP_LOW = np.array([0.0, 0.0, 0.5], dtype=np.float32) # Dark Blue
COLORMAP_HIGH = np.array([0.8, 0.8, 0.2], dtype=np.float32) # Yellowish
# --- Globals ---
height_data = None
vertices = None
normals = None
colors = None
indices = None
# vao = None # Removed VAO
vbo = None
ibo = None
shader_program = None

# Rotation/Camera variables
angle_y = 0.0
zoom = -2000.0 # Initial distance (adjust based on terrain size) - Increased initial distance
last_mouse_x = -1
last_mouse_y = -1
mouse_dragging = False

# --- Shader Code (GLSL 1.20 for OpenGL 2.1 Compatibility) ---
VERTEX_SHADER = """
#version 120

attribute vec3 aPos;    // Replaced 'layout (location = 0) in'
attribute vec3 aNormal; // Replaced 'layout (location = 1) in'
attribute vec3 aColor;  // Replaced 'layout (location = 2) in'

varying vec3 FragPos;   // Replaced 'out'
varying vec3 Normal;    // Replaced 'out'
varying vec3 Color;     // Replaced 'out'

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
// GLSL 1.20 doesn't have transpose/inverse directly in mat3 constructor often
// Let's calculate the normal matrix on the CPU if needed, or use gl_NormalMatrix if safe
// For simplicity here, we keep the potentially expensive calculation,
// but be aware gl_NormalMatrix exists in compatibility profiles.
// uniform mat3 normalMatrix; // Alternative: Pass pre-calculated normal matrix

void main()
{
    // Calculate FragPos in World Space
    FragPos = vec3(model * vec4(aPos, 1.0));

    // Transform Normal: Calculate normal matrix (inverse transpose of model's upper 3x3)
    // This explicit calculation works but is less efficient than built-ins.
    // Note: inverse() might not be available on all GLSL 1.20 implementations.
    // A common approximation is to assume uniform scaling and just use mat3(model).
    // For non-uniform scaling, the full inverse transpose is needed.
    // Let's try the simpler version first assuming no non-uniform scaling in the model matrix:
    Normal = normalize(mat3(model) * aNormal); // Use simpler version
    // If non-uniform scaling occurs, use the more complex method or pass from CPU
    // mat3 normalMatrix = mat3(transpose(inverse(mat3(model)))); // Keep original logic for accuracy - Causes GLSL 1.20 error
    // Normal = normalize(normalMatrix * aNormal); // Use the line above instead

    Color = aColor;
    // Calculate final position in Clip Space
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 120

varying vec3 FragPos;   // Replaced 'in'
varying vec3 Normal;    // Replaced 'in'
varying vec3 Color;     // Replaced 'in'

uniform vec3 lightDir; // Direction *towards* the light source
uniform vec3 lightColor;
// uniform vec3 viewPos; // viewPos isn't used in this basic diffuse shader
uniform bool useLighting;

void main()
{
    if (useLighting) {
        float ambientStrength = 0.2;
        vec3 ambient = ambientStrength * lightColor;

        // Diffuse lighting
        vec3 norm = normalize(Normal);
        vec3 lightDirection = normalize(lightDir); // Ensure lightDir is normalized
        float diff = max(dot(norm, lightDirection), 0.0);
        vec3 diffuse = diff * lightColor;

        // Combine ambient and diffuse, modulated by vertex color
        vec3 result = (ambient + diffuse) * Color;
        gl_FragColor = vec4(result, 1.0); // Replaced FragColor = ...
    } else {
        gl_FragColor = vec4(Color, 1.0); // Replaced FragColor = ... No lighting, just vertex color
    }
}
"""

# --- Helper Functions ---

def load_heightmap(filepath):
    """Loads heightmap, returns NumPy array."""
    global height_data
    try:
        img = Image.open(filepath)
        img_gray = img.convert('L')
        print(f"Image loaded: {filepath} (Original Mode: {img.mode}, Size: {img.size})")
        print(f"Converted to Grayscale. Min/Max pixel values: {img_gray.getextrema()}")
        height_data = np.array(img_gray, dtype=np.float32)

        if INVERT_HEIGHTMAP:
            print("Inverting heightmap data (255 - value).")
            height_data = 255.0 - height_data

        # Downsample
        if DOWNSAMPLE_FACTOR > 1:
            ds = int(DOWNSAMPLE_FACTOR)
            height_data = height_data[::ds, ::ds]
            print(f"Downsampled height data to: {height_data.shape}")

        if height_data.size == 0 or height_data.shape[0] <= 1 or height_data.shape[1] <= 1:
             print("Error: Invalid height data dimensions after loading/downsampling.")
             sys.exit(1)

        return True
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return False
    except Exception as e:
        print(f"Error loading image: {e}")
        return False

def create_mesh_data():
    """Generates vertices, normals, colors, and indices from height_data."""
    global vertices, normals, colors, indices
    if height_data is None:
        return False

    rows, cols = height_data.shape
    max_height = 255.0 # Assuming original image range
    scale_factor = max(rows, cols) / max_height * VERTICAL_EXAGGERATION

    verts = []
    norms = []
    cols_array = []
    inds = []

    # Center the mesh around origin
    x_offset = -cols / 2.0
    y_offset = -rows / 2.0

    # Temporary storage for normal calculation
    temp_normals = {}

    for r in range(rows):
        for c in range(cols):
            # --- Vertex Position ---
            x = float(c) + x_offset
            y = float(r) + y_offset
            z = (height_data[r, c]) * scale_factor
            verts.extend([x, y, z])

            # --- Vertex Color (Simple Lerp based on normalized height) ---
            normalized_height = height_data[r, c] / max_height
            color = COLORMAP_LOW * (1.0 - normalized_height) + COLORMAP_HIGH * normalized_height
            cols_array.extend(color)

            # --- Calculate Normals (Approximate using finite differences) ---
            # We need neighbors, calculate later in a separate pass for simplicity here
            # Store placeholder normal for now
            norms.extend([0.0, 0.0, 1.0]) # Default up

    # --- Calculate Normals (Second Pass) ---
    # Simple method: average normals of triangles surrounding a vertex
    # More accurate: use cross products of vectors to neighbors
    temp_norms_list = [[0.0, 0.0, 0.0] for _ in range(rows * cols)]

    for r in range(rows - 1):
        for c in range(cols - 1):
            idx00 = r * cols + c
            idx10 = (r + 1) * cols + c
            idx01 = r * cols + (c + 1)
            idx11 = (r + 1) * cols + (c + 1)

            # Get vertex positions for the quad
            v00 = np.array(verts[idx00*3 : idx00*3+3])
            v10 = np.array(verts[idx10*3 : idx10*3+3])
            v01 = np.array(verts[idx01*3 : idx01*3+3])
            v11 = np.array(verts[idx11*3 : idx11*3+3])

            # Triangle 1: (00, 10, 01)
            vecA = v10 - v00
            vecB = v01 - v00
            norm1 = np.cross(vecA, vecB)

            # Triangle 2: (10, 11, 01)
            vecC = v11 - v10
            vecD = v01 - v10
            norm2 = np.cross(vecC, vecD)

            # Accumulate normals (non-normalized)
            temp_norms_list[idx00] += norm1
            temp_norms_list[idx10] += norm1 + norm2
            temp_norms_list[idx01] += norm1 + norm2
            temp_norms_list[idx11] += norm2

    # Normalize accumulated normals and update the 'norms' list
    norms = []
    for n in temp_norms_list:
        norm_vec = np.array(n)
        mag = np.linalg.norm(norm_vec)
        if mag > 0.0001:
            norm_vec /= mag
        else:
            norm_vec = np.array([0.0, 0.0, 1.0]) # Default up if magnitude is zero
        norms.extend(norm_vec)


    # --- Generate Indices for Triangles ---
    for r in range(rows - 1):
        for c in range(cols - 1):
            idx00 = r * cols + c
            idx10 = (r + 1) * cols + c
            idx01 = r * cols + (c + 1)
            idx11 = (r + 1) * cols + (c + 1)
            # Triangle 1
            inds.extend([idx00, idx10, idx01])
            # Triangle 2
            inds.extend([idx10, idx11, idx01])

    # Convert lists to NumPy arrays
    vertices = np.array(verts, dtype=np.float32)
    normals = np.array(norms, dtype=np.float32) # Add normals here
    colors = np.array(cols_array, dtype=np.float32)
    indices = np.array(inds, dtype=np.uint32)

    print(f"Mesh created: {len(vertices)//3} vertices, {len(indices)//3} triangles.")
    return True

def compile_shader(source, shader_type):
    """Compiles shader source and returns shader ID."""
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        info = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compilation failed: {info}")
    return shader

def link_program(vertex_shader, fragment_shader):
    """Links shaders into a program and returns program ID."""
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        info = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Shader program linking failed: {info}")
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program

def numpy_to_mat4(np_mat):
    """Converts a NumPy 4x4 matrix to a flat list for PyOpenGL."""
    return np_mat.flatten('F') # Use Fortran order for OpenGL

def create_view_matrix(eye, target, up):
    """Creates a look-at view matrix (like gluLookAt)."""
    f = np.array(target) - np.array(eye)
    f = f / np.linalg.norm(f)
    s = np.cross(f, np.array(up))
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    mat = np.identity(4, dtype=np.float32)
    mat[0, 0] = s[0]
    mat[1, 0] = s[1]
    mat[2, 0] = s[2]
    mat[0, 1] = u[0]
    mat[1, 1] = u[1]
    mat[2, 1] = u[2]
    mat[0, 2] = -f[0]
    mat[1, 2] = -f[1]
    mat[2, 2] = -f[2]
    mat[3, 0] = -np.dot(s, eye)
    mat[3, 1] = -np.dot(u, eye)
    mat[3, 2] = np.dot(f, eye)
    return mat

def create_projection_matrix(fov, aspect, near, far):
    """Creates a perspective projection matrix (like gluPerspective)."""
    f = 1.0 / math.tan(math.radians(fov) / 2.0)
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2.0 * far * near) / (near - far)
    mat[3, 2] = -1.0
    return mat

# --- OpenGL Initialization ---
def init_gl():
    """Initializes OpenGL state, shaders, and buffers."""
    # global vao, vbo, ibo, shader_program # Removed vao from globals
    global vbo, ibo, shader_program

    glClearColor(0.1, 0.1, 0.2, 1.0) # Dark blue background
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE) # Re-enable culling
    glCullFace(GL_BACK)    # Cull back faces
    glFrontFace(GL_CW)     # Assume clockwise triangles are front-facing

    # Compile shaders
    try:
        vs = compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)
        fs = compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    except RuntimeError as e:
        print(e)
        sys.exit(1) # Exit if compilation fails

    # Create shader program *before* linking
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vs)
    glAttachShader(shader_program, fs)

    # --- Explicitly bind attribute locations BEFORE linking ---
    # These names ("aPos", "aNormal", "aColor") MUST match the 'attribute'
    # variable names in the VERTEX_SHADER.
    # The indices (0, 1, 2) correspond to the locations used in glVertexAttribPointer.
    glBindAttribLocation(shader_program, 0, "aPos")
    glBindAttribLocation(shader_program, 1, "aNormal")
    glBindAttribLocation(shader_program, 2, "aColor")
    # --- End binding ---

    # Link the program
    glLinkProgram(shader_program)
    if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
        info = glGetProgramInfoLog(shader_program).decode()
        # Clean up shaders even on failure before raising error
        glDeleteShader(vs)
        glDeleteShader(fs)
        glDeleteProgram(shader_program) # Clean up program object
        raise RuntimeError(f"Shader program linking failed: {info}")

    # Delete shaders after successful linking (they are linked into the program)
    glDeleteShader(vs)
    glDeleteShader(fs)

    # --- Create VAO, VBO, IBO ---
    # VAOs are technically core OpenGL 3.0+, but often available as extensions
    # on 2.1 contexts. If VAOs cause issues, you might need to revert to
    # binding VBOs and setting up attribute pointers *every frame* before drawing.
    # --- VAO Removed ---
    # vao = glGenVertexArrays(1) # Causes GL_INVALID_OPERATION on some 2.1 contexts
    # glBindVertexArray(vao)     # Removed

    # Combine vertex data (pos, normal, color) into one buffer
    vertex_data = np.hstack((vertices.reshape(-1, 3),
                             normals.reshape(-1, 3),
                             colors.reshape(-1, 3))).flatten()

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

    ibo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # --- Configure Vertex Attributes ---
    # The locations (0, 1, 2) now match the glBindAttribLocation calls
    stride = 9 * vertices.itemsize # Stride in bytes (3 pos + 3 norm + 3 color) * float_size

    # Position attribute (location=0)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

    # Normal attribute (location=1)
    glEnableVertexAttribArray(1)
    offset_normal = 3 * vertices.itemsize # Offset in bytes (after 3 position floats)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset_normal))

    # Color attribute (location=2)
    glEnableVertexAttribArray(2)
    offset_color = 6 * vertices.itemsize # Offset in bytes (after 3 pos + 3 norm floats)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset_color))

    # --- VAO Removed ---
    # Unbind VBO only (VAO is gone, IBO remains bound implicitly by context until needed)
    # glBindVertexArray(0) # Removed
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    # Don't unbind IBO here, it's needed for drawing later

    # Check for GL errors after setup
    err = glGetError()
    if err != GL_NO_ERROR:
        print(f"OpenGL Error after init_gl: {err}") # Consider using gluErrorString(err) if available

# --- GLUT Callbacks ---
def display():
    """GLUT display callback."""
    global angle_y
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader_program)

    # --- Transformations ---
    # Model Matrix (Rotation)
    rad_y = math.radians(angle_y)
    cos_y = math.cos(rad_y)
    sin_y = math.sin(rad_y)
    model_mat = np.array([
        [cos_y, 0, sin_y, 0],
        [0,     1, 0,     0],
        [-sin_y,0, cos_y, 0],
        [0,     0, 0,     1]
    ], dtype=np.float32)

    # View Matrix (Camera) - Simple orbit around origin
    cam_x = 0 # math.sin(math.radians(camera_angle_x)) * abs(zoom)
    # cam_y = 150 # Fixed height above center (adjust as needed) - Original value
    cam_y = 1500 # Keep increased height
    cam_z = zoom # Use the zoom variable again for scroll control
    eye = (cam_x, cam_y, cam_z)
    target = (0, 0, 0) # Look at the center of the terrain
    up = (0, 1, 0)
    view_mat = create_view_matrix(eye, target, up)

    # Projection Matrix
    aspect = float(WINDOW_WIDTH) / float(WINDOW_HEIGHT)
    projection_mat = create_projection_matrix(45.0, aspect, 10.0, 5000.0) # Adjust near/far

    # --- Upload Uniforms ---
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, numpy_to_mat4(model_mat))
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, numpy_to_mat4(view_mat))
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, numpy_to_mat4(projection_mat))

    # Lighting uniforms
    light_dir = np.array([0.8, 0.6, 0.5], dtype=np.float32) # Directional light source direction
    light_dir /= np.linalg.norm(light_dir) # Normalize
    light_col = np.array([1.0, 1.0, 1.0], dtype=np.float32) # White light

    glUniform3fv(glGetUniformLocation(shader_program, "lightDir"), 1, light_dir)
    glUniform3fv(glGetUniformLocation(shader_program, "lightColor"), 1, light_col)
    glUniform3fv(glGetUniformLocation(shader_program, "viewPos"), 1, np.array(eye, dtype=np.float32)) # Pass camera position
    glUniform1i(glGetUniformLocation(shader_program, "useLighting"), USE_LIGHTING)

    # --- Draw (No VAO) ---
    # Bind VBO and set attribute pointers before drawing
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo) # Bind IBO for glDrawElements

    stride = 9 * vertices.itemsize # Stride in bytes (3 pos + 3 norm + 3 color) * float_size
    # Position attribute (location=0)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    # Normal attribute (location=1)
    glEnableVertexAttribArray(1)
    offset_normal = 3 * vertices.itemsize
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset_normal))
    # Color attribute (location=2)
    glEnableVertexAttribArray(2)
    offset_color = 6 * vertices.itemsize
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset_color))

    # Actual draw call
    # glBindVertexArray(vao) # Removed
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
    # glBindVertexArray(0) # Removed

    # Disable attribute arrays and unbind buffers after drawing
    glDisableVertexAttribArray(0)
    glDisableVertexAttribArray(1)
    glDisableVertexAttribArray(2)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)


    glUseProgram(0) # Unbind shader program
    glutSwapBuffers()

def reshape(w, h):
    """GLUT reshape callback."""
    global WINDOW_WIDTH, WINDOW_HEIGHT
    if h == 0: h = 1 # Prevent divide by zero
    WINDOW_WIDTH, WINDOW_HEIGHT = w, h
    glViewport(0, 0, w, h)
    # Projection matrix update is handled in display() based on current width/height

def keyboard(key, x, y):
    """GLUT keyboard callback."""
    global USE_LIGHTING
    key_code = key.decode("utf-8").lower()
    if key_code == '\x1b' or key_code == 'q': # Escape or Q key
        print("Exiting...")
        # Cleanup (optional, OS usually handles it)
        # glDeleteProgram(shader_program)
        # glDeleteVertexArrays(1, [vao]) # Removed VAO cleanup
        # glDeleteBuffers(1, [vbo])
        # glDeleteBuffers(1, [ibo])
        glutLeaveMainLoop()
    elif key_code == 'l':
        USE_LIGHTING = not USE_LIGHTING
        print(f"Lighting {'Enabled' if USE_LIGHTING else 'Disabled'}")
        glutPostRedisplay()

def mouse_button(button, state, x, y):
    """GLUT mouse button callback."""
    # Declare globals you intend to modify *first*
    global mouse_dragging, last_mouse_x, last_mouse_y, zoom

    if button == GLUT_LEFT_BUTTON:
        if state == GLUT_DOWN:
            mouse_dragging = True
            last_mouse_x, last_mouse_y = x, y
        elif state == GLUT_UP:
            mouse_dragging = False
    elif button == 3: # Scroll wheel up (Zoom In)
         # No need for global declaration here anymore
         zoom += 50.0 # Closer: Increase the negative value (e.g. -500 -> -450)
         # Keep zoom from getting too close or positive
         zoom = min(-10.0, zoom) # Ensure it stays negative and not too close
         print(f"Zoom In: {zoom}")
         glutPostRedisplay()
    elif button == 4: # Scroll wheel down (Zoom Out)
         # No need for global declaration here anymore
         zoom -= 50.0 # Further: Decrease the negative value (e.g. -500 -> -550)
         print(f"Zoom Out: {zoom}")
         glutPostRedisplay()


def mouse_motion(x, y):
    """GLUT mouse motion callback (for dragging)."""
    global last_mouse_x, last_mouse_y, angle_y
    if mouse_dragging:
        dx = x - last_mouse_x
        # dy = y - last_mouse_y # Could use dy for vertical rotation/pitch

        angle_y += dx * ROTATION_SPEED
        # camera_angle_x += dy * ROTATION_SPEED # Add if implementing pitch

        last_mouse_x, last_mouse_y = x, y
        glutPostRedisplay()

def idle():
    """GLUT idle callback (can be used for continuous animation)."""
    # Uncomment for automatic rotation
    # global angle_y
    # angle_y = (angle_y + ROTATION_SPEED) % 360
    # glutPostRedisplay()
    pass

# --- Main Execution ---
def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(__file__)} <path_to_heightmap_image>")
        return

    filepath = sys.argv[1]
    if not load_heightmap(filepath):
        return
    if not create_mesh_data():
        return

    # Initialize GLUT
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA)
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT)
    glutCreateWindow(b"OpenGL Terrain Viewer") # Needs bytes string

    print("OpenGL Initialized. Version:", glGetString(GL_VERSION).decode())
    print("GLSL Version:", glGetString(GL_SHADING_LANGUAGE_VERSION).decode())

    # Initialize OpenGL state, shaders, buffers
    init_gl()

    # Register callbacks
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse_button)
    glutMotionFunc(mouse_motion)
    # glutIdleFunc(idle) # Uncomment for continuous animation

    print("\nControls:")
    print("  Mouse Drag Left: Rotate")
    print("  Mouse Scroll: Zoom")
    print("  L Key: Toggle Lighting")
    print("  Q or Esc: Quit")

    # Enter GLUT main loop
    glutMainLoop()

if __name__ == "__main__":
    main()
