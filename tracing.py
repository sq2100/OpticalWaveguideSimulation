import taichi as ti
import taichi.math as tm
import pygmsh

ti.init(arch=ti.gpu,default_fp=ti.f64)
image_resolution = (1000, 1000)

# 光波导参数
wave_points = 30 # 离散数
wave_half_high = 0.01
path_num = 500

# 配置常量
TMIN        = 0.001                 # 光开始传播的起始偏移，避免光线自相交
TMAX        = 2000.0                # 最大单次光线传播距离
PRECISION   = 0.0001                # 必须要小于 TMIN，否则光线会自相交产生阴影痤疮
MAX_RAYMARCH = 512                  # 最大光线步进次数
MAX_RAYTRACE = 512  # 最大路径追踪次数
CLAD_IOR = 1.41  # 包层的折射率
WAV_IOR = 1.48  # 光波导折射率


# 初始化几何
with pygmsh.geo.Geometry() as geom:
    outer_polygon = geom.add_polygon(
        [
            [-0.5,-0.2],
            [0.5, -0.2],
            [0.5, 0.2],
            [-0.5, 0.2],
        ],
        mesh_size=0.1
    )
    mesh = geom.generate_mesh()
nodes = mesh.points
mesh_triangles = mesh.cells_dict['triangle']
# print(nodes.shape)
nodes_x = ti.Vector.field(3, dtype=ti.f64, shape=(nodes.shape[0],))
indices = ti.field(int, shape=(mesh_triangles.shape[0]*mesh_triangles.shape[1],))
curve_points = ti.Vector.field(3, dtype=ti.f64, shape=wave_points)
light_pow = ti.field(dtype=ti.f32, shape=())
indices_line = ti.field(int, shape=(wave_points*2-2,))





P = ti.Vector.field(2, dtype=ti.f64, shape=4) # 控制点
t = ti.field(ti.f64, shape=wave_points) # 插值点
B = ti.Vector.field(2, dtype=ti.f64, shape=wave_points) # 二维贝塞尔点

nodes_x.from_numpy(nodes)
indices.from_numpy(mesh_triangles.flatten())


@ti.dataclass
class Ray:  # 光线类
    origin: tm.vec3    # 光线起点
    direction: tm.vec3 # 光线方向
    color: tm.vec4     # 光的颜色

    @ti.func
    def at(self, t: float) -> tm.vec3: # 计算光子所在位置
        return self.origin + t * self.direction
    
@ti.dataclass
class HitRecord:    # 光子碰撞记录类
    position: tm.vec3  # 光子碰撞的位置
    distance: float # 光子步进的距离
    hit: ti.i32     # 是否击中到物体
    hit_light: ti.i32     # 是否击中到物体

@ti.kernel
def init_waveguide():
    P[0] = [-0.5, 0.0]
    P[1] = [-0.2, 0.1]
    P[2] = [0.2, 0]
    P[3] = [0.5, 0.0]
    for i in curve_points:
        t[i] = float(i) / (wave_points - 1)
        B[i] = (1 - t[i])**3 * P[0] + 3 * (1 - t[i])**2 * t[i] * P[1] + 3 * (1 - t[i]) * t[i]**2 * P[2] + t[i]**3 * P[3]
        curve_points[i] = ti.Vector([B[i][0], B[i][1], 0.0])
    for i in range(curve_points.shape[0]-1):
        indices_line[i*2], indices_line[i*2+1] = i, i+1

@ti.func
def sd_wave(p: tm.vec3, r: float = wave_half_high) -> float:
    min_dist = float('inf')
    for i in ti.static(range(wave_points - 1)):
        a = curve_points[i]
        b = curve_points[i + 1]
        a[1] += r  # 上平移
        b[1] += r
        t = ti.max(0.0, ti.min(1.0, (p - a).dot(b - a) / (b - a).norm_sqr()))
        projection = a + t * (b - a)
        d1 = (p - projection).norm()

        a[1] -= 2*r  # 下平移
        b[1] -= 2*r
        t = ti.max(0.0, ti.min(1.0, (p - a).dot(b - a) / (b - a).norm_sqr()))
        projection = a + t * (b - a)
        d2 = (p - projection).norm()

        min_dist = ti.min(min_dist, ti.min(d1, d2))
    return min_dist

@ti.func
def sd_li(p: tm.vec3, r: float = wave_half_high) -> float:
    min_dist = float('inf')
    a = tm.vec3(P[3][0], P[3][1], 0.0)
    b = a
    a[1] += r  
    b[1] -= r
    t = ti.max(0.0, ti.min(1.0, (p - a).dot(b - a) / (b - a).norm_sqr()))
    projection = a + t * (b - a)
    d = (p - projection).norm()
    return d

@ti.func
def calc_normal(p: tm.vec3, r: float = wave_half_high) -> tm.vec3:
    min_dist = float('inf')
    normal = tm.vec3(0.0, 0.0, 0.0)
    
    for i in ti.static(range(wave_points - 1)):
        a = curve_points[i]
        b = curve_points[i + 1]
        a[1] += r  # 上平移
        b[1] += r
        t = ti.max(0.0, ti.min(1.0, (p - a).dot(b - a) / (b - a).norm_sqr()))
        projection = a + t * (b - a)
        d1 = (p - projection).norm()
        
        a[1] -= 2*r  # 下平移
        b[1] -= 2*r
        t = ti.max(0.0, ti.min(1.0, (p - a).dot(b - a) / (b - a).norm_sqr()))
        projection = a + t * (b - a)
        d2 = (p - projection).norm()

        if ti.min(d1, d2) < min_dist:
            min_dist = ti.min(d1, d2)
            tangent = (b - a).normalized()
            normal = tm.vec3(-tangent.y, tangent.x, 0.0)  # 旋转90度得到法线

    return normal

@ti.func
def raycast(ray) -> HitRecord:  # 光线步进求交
    record = HitRecord(ray.origin, TMIN, False) # 初始化光子碰撞记录
    i=0
    for _ in range(MAX_RAYMARCH):   # 光线步进
        record.position = ray.at(record.distance)   # 计算光子所在位置
        # 计算光子与光波导的sd
        sd_waveguide = sd_wave(record.position)
        sd_light = sd_li(record.position)
        dis_light = ti.abs(sd_light)
        dis_waveguide = ti.abs(sd_waveguide)
        dis = ti.min(dis_light, dis_waveguide)
        if dis_light < PRECISION: # 如果光子与球体的距离小于精度即为击中
            record.hit = True   # 设置击中状态
            record.hit_light = True
            # print(record.position,'++++',i,'$$$',dis_light,'&&&',dis_waveguide,'::hit:',record.hit, 'lig:',record.hit_light,' dis:',record.distance)
            break
        if dis < PRECISION: # 如果光子与球体的距离小于精度即为击中
            record.hit = True   # 设置击中状态
            # print(record.position,'----',i,'$$$',dis_light,'&&&',dis_waveguide,'::hit:',record.hit, 'lig:',record.hit_light,' dis:',record.distance)
            break
        record.distance += dis  # 光子继续传播
        # print(record.position,'----',i,'$$$',dis_light,'&&&',dis_waveguide,'::hit:',record.hit, 'lig:',record.hit_light,' dis:',record.distance)
        if record.distance > TMAX:  # 如果光子传播距离大于最大传播距离
            break
        i+=1
    # print('or:',ray.origin,'--',i,'::hit:',record.hit, 'lig:',record.hit_light,' dis:',record.distance)
    return record   # 返回光子碰撞记录

@ti.kernel
def tracing_waveguides(): # 计算光强
    light_pow[None] = 0
    for i in ti.ndrange(path_num):
        y_pos = (ti.random() - 0.5) * wave_half_high * 2
        origin = tm.vec3(-0.5, y_pos, 0)      # 发射点
        ro = origin # 光线起点
        rd = tm.vec3(1,0,0)
        color = tm.vec4(1.0)
        ray = Ray(ro, rd, color)   # 创建光线
        ray = raytrace(ray) # 光线追踪
        light_pow[None] += ray.color.a
        # print(i,':',ray.color,' ypos:',y_pos)
    light_pow[None] = light_pow[None]/path_num
    # print("light_pow",light_pow)

@ti.func
def pow5(x: float): # 快速计算 x 的 5 次方
    t = x*x
    t *= t
    return t*x

@ti.func
def fresnel_schlick(cosine: float, F0: float) -> float:   # 计算菲涅尔近似值
    return F0 + (1 - F0) * pow5(1 - abs(cosine))

@ti.func
def raytrace(ray) -> Ray: # 循环光线
    for _ in range(MAX_RAYTRACE):
        record = raycast(ray)   # 光线步进求交
        if not record.hit:
            ray.color.rgb *= tm.vec3(0,0,0)
            break

        if record.hit_light:
            ray.color.a *= 100
            break

        visible = tm.length(ray.color.rgb*ray.color.a)
        if visible < 0.001:  # 如果光已经衰减到不可分辨程度了就不继续了
            break
        
        # 这里的法线会始终指向物体外面
        normal = calc_normal(record.position) # 计算法线
        # ray.direction = tm.reflect(ray.direction, normal)  # 反射光线
        # ray.origin = record.position    # 设置光线起点
        I = ray.direction   # 入射方向
        V = -ray.direction  # 观察方向
        NoV = tm.dot(normal, V) # sin

        eta = WAV_IOR/CLAD_IOR  # 折射率之比

        NoI = -NoV # 入射光线方向（I）与表面法线（N）的点积
        k = 1.0 - eta * eta * (1.0 - NoI * NoI) # 这个值如果小于 0 就说明全反射了
        F0 = (eta - 1) / (eta + 1)  # 基础反射率
        F0 *= F0
        F = fresnel_schlick(NoV, F0)
        # print(":::::",I,NoV,F0,F)
        if ti.random() < F or k < 0:
            ray.direction = tm.reflect(ray.direction, normal)  # 反射
            ray.origin = record.position    # 设置光线起点
        else:
            break

        # ray.color.a *= 0.5  # 让强度衰减一些
    return ray

