import taichi as ti
from tracing import *
 
# global control
paused = True
curser = ti.Vector.field(2, dtype=float, shape=(1, ))
curser[0] = [0, 0]
curser_3d = ti.Vector.field(3, dtype=float, shape=(1, ))
curser_3d[0] = [0,0,0]
picking = ti.field(ti.i32,())

# physical quantities
m = 1
g = 9.8
YoungsModulus = ti.field(ti.f32, ())
PoissonsRatio = ti.field(ti.f32, ())
LameMu = ti.field(ti.f32, ())
LameLa = ti.field(ti.f32, ())
curser_radius = 0.1

alpha = 0.1  # Mass-proportional damping coefficient
beta = 0.2  # Stiffness-proportional damping coefficient

# time-step size (for simulation, 16.7ms)
h = 16.7e-3
# substepping
substepping = 100
# time-step size (for time integration)
dh = h/substepping

N = nodes.shape[0]
N_triangles = mesh_triangles.shape[0]

# simulation components
x = ti.Vector.field(2, ti.f32, N, needs_grad=True)
v = ti.Vector.field(2, ti.f32, N)
total_energy = ti.field(ti.f32, (), needs_grad=True)
grad = ti.Vector.field(2, ti.f32, N)
elements_Dm_inv = ti.Matrix.field(2, 2, ti.f32, N_triangles)
elements_V0 = ti.field(ti.f32, N_triangles)
x_grad = ti.Vector.field(2, ti.f32, N)  # new field to store the gradient of total energy


# geometric components
triangles = ti.Vector.field(3, ti.i32, N_triangles)
# edges = ti.Vector.field(2, ti.i32, N_edges)
triangles.from_numpy(mesh_triangles)

curve_triangle_indices = ti.field(dtype=ti.i32, shape=wave_points)
barycentric_coords = ti.Vector.field(3, dtype=ti.f32, shape=wave_points)

@ti.kernel
def find_triangle_and_barycentric():
    # 遍历所有标记点
    for i in range(wave_points):
        # 遍历所有三角形
        for j in range(N_triangles):
            # 获取三角形的顶点
            triangle = triangles[j]
            A = x[triangle[0]]
            B = x[triangle[1]]
            C = x[triangle[2]]
            P = curve_points[i]
            
            # 计算面积
            ABC = abs(A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1]))
            ABP = abs(A[0]*(B[1]-P[1]) + B[0]*(P[1]-A[1]) + P[0]*(A[1]-B[1]))
            BCP = abs(B[0]*(C[1]-P[1]) + C[0]*(P[1]-B[1]) + P[0]*(B[1]-C[1]))
            CAP = abs(C[0]*(A[1]-P[1]) + A[0]*(P[1]-C[1]) + P[0]*(C[1]-A[1]))
            
            # 如果点在三角形内，更新 curve_triangle_indices[i] 和 barycentric_coords[i]
            if abs(ABC - (ABP + BCP + CAP)) < 1e-5:
                curve_triangle_indices[i] = j
                w_A = BCP / ABC
                w_B = CAP / ABC
                w_C = ABP / ABC
                barycentric_coords[i] = ti.Vector([w_A, w_B, w_C])
                break



@ti.kernel
def initialize_fem():
    YoungsModulus[None] = 60000
    paused = True
    # init position and velocity
    for i in nodes_x:
        v[i] = tm.vec2(0.0,0.0)
        x[i][0] = nodes_x[i][0]
        x[i][1] = nodes_x[i][1]


@ti.kernel
def initialize_elements():
    for i in range(N_triangles):
        Dm = compute_D(i)
        elements_Dm_inv[i] = Dm.inverse()
        elements_V0[i] = ti.abs(Dm.determinant())/2

@ti.func
def compute_D(i):
    a = triangles[i][0]
    b = triangles[i][1]
    c = triangles[i][2]
    return ti.Matrix.cols([x[b] - x[a], x[c] - x[a]])

@ti.func
def compute_R_2D(F):
    R, S = ti.polar_decompose(F, ti.f32)
    return R


@ti.kernel
def compute_total_energy():
    for i in range(N_triangles):
        Ds = compute_D(i)
        F = Ds @ elements_Dm_inv[i]
        # co-rotated linear elasticity
        R = compute_R_2D(F)
        Eye = ti.Matrix.cols([[1.0, 0.0], [0.0, 1.0]])
        element_energy_density = LameMu[None]*((F-R)@(F-R).transpose()).trace() + 0.5*LameLa[None]*(R.transpose()@F-Eye).trace()**2
        total_energy[None] += element_energy_density * elements_V0[i]   

@ti.kernel
def updateLameCoeff():
    E = YoungsModulus[None]
    nu = PoissonsRatio[None]
    LameLa[None] = E*nu / ((1+nu)*(1-2*nu))
    LameMu[None] = E / (2*(1+nu))


@ti.kernel
def update():
    # perform time integration
    for i in range(N):
        # symplectic integration
        # # elastic force + gravitation force, divding mass to get the acceleration
        # damping_force_stiffness = beta * x_grad[i]

        # # Compute the mass-proportional damping force
        # damping_force_mass = alpha * v[i]

        # # Compute the total damping force
        # damping_force = damping_force_mass + damping_force_stiffness

        # # symplectic integration
        # # elastic force + gravitation force + damping force, dividing mass to get the acceleration
        # acc = -(x.grad[i] + damping_force) / m - ti.Vector([0.0, g])

        acc = -x.grad[i]/m - ti.Vector([0.0, g])
        v[i] += dh*acc
        x[i] += dh*v[i]

        v[i] *= ti.exp(-dh*50)

    for i in range(N):
        if x[i][1] <= -0.18:
            x[i][1] = -0.2
            v[i] = tm.vec2(0.0,0.0)

    for i in nodes_x:
        nodes_x[i][0] = x[i][0]
        nodes_x[i][1] = x[i][1]

    for i in range(N):
        if picking[None]:           
            r = x[i] - curser[0]
            distance = r.norm()
            if distance < curser_radius:
                # Velocity projection
                normal = r.normalized()
                v[i] -= ti.min(v[i].dot(normal), 0) * normal
                # Position update: move the particle to the surface of the sphere
                x[i] = curser[0] + normal * curser_radius


    for i in curve_points:
        # 获取标记点所在的三角形索引和重心坐标
        triangle_index = curve_triangle_indices[i]
        barycentric_coord = barycentric_coords[i]
        
        # 获取三角形的顶点索引
        a, b, c = triangles[triangle_index]

        # 使用重心坐标和新的节点坐标计算新的标记点坐标
        new_point = x[a] * barycentric_coord[0] + x[b] * barycentric_coord[1] + x[c] * barycentric_coord[2]
        
        # 更新标记点坐标
        curve_points[i] = tm.vec3(new_point[0],new_point[1],0.0)


init_waveguide()
initialize_fem()
initialize_elements()
updateLameCoeff()
find_triangle_and_barycentric()

# 可视化
window = ti.ui.Window("Simulation", image_resolution, vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
gui = window.get_gui()

while window.running:
    # print(dir(ti.ui.ProjectionMode))
    camera.projection_mode(ti.ui.ProjectionMode.Orthogonal)
    camera.position(0.0, 0.0, 1)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    picking[None]=0

    
    if window.is_pressed(ti.ui.LMB):
        mouse_x, mouse_y = window.get_cursor_pos()
        mouse_x = (mouse_x - 0.5) * 2
        mouse_y = (mouse_y - 0.5) * 2
        curser[0] = [mouse_x, mouse_y]  
        # print(window.get_cursor_pos(),":::",curser[0])
        curser_3d[0][0] = curser[0][0]
        curser_3d[0][1] = curser[0][1]
        picking[None] = 1
        scene.particles(curser_3d, radius=curser_radius*0.4, color=(0.4784, 0.5804, 0.8039))
        scene.mesh(nodes_x,
               indices=indices,
               color=(0.5, 0.5, 0.5))

    for i in range(substepping):
        total_energy[None]=0
        with ti.ad.Tape(total_energy):
            compute_total_energy()
        # for i in range(N):
        #     x_grad[i] = x.grad[i]
        update()


    tracing_waveguides()
    with gui.sub_window("Sub Window", x=0.1, y=0.1, width=0.2, height=0.1):
        gui.text("LIGHT_POW:"+str(light_pow[None]))
        # P[1][1] = gui.slider_float("Col_p1", P[1][1], minimum=-0.2, maximum=0.2)
        # P[2][1] = gui.slider_float("Col_p2", P[2][1], minimum=-0.2, maximum=0.2)

    
    scene.mesh(nodes_x,
               indices=indices,
               color=(0.5, 0.5, 0.5))
    scene.particles(nodes_x, radius=0.005, color=(0.5, 0.5, 0.5)) # 有限元顶点
    scene.lines(curve_points, indices=indices_line ,width=image_resolution[1]*wave_half_high*2, color=(0.8039, 0.1608, 0.1059))
    scene.particles(curve_points, radius=0.002, color=(0.4784, 0.5804, 0.8039))
    
    canvas.scene(scene)
    window.show()