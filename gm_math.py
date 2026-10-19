# Replica exacta de las matrices de GameMaker usadas por Mine-imator.
#
# GameMaker usa vectores fila (v' = v * M) y matrices de 16 floats donde la
# traslacion esta en los indices 12, 13, 14. matrix_multiply(A, B) aplica
# primero A y luego B. matrix_build esta portado literalmente del runner
# de GameMaker (YoYoGames/GameMaker-HTML5, Matrix.js -> BuildMatrix), incluidas
# las rotaciones negadas y el orden de escala.

import math

IDENTITY = (1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0)


def build(x, y, z, xrot, yrot, zrot, xs, ys, zs):
    """matrix_build de GameMaker: la escala se aplica despues de rotar, en los
    ejes del padre (columnas escaladas). Comprobado con proyectos reales: los
    rigs compensan la escala no uniforme de un padre con la del hijo."""
    p = math.radians(-xrot)
    h = math.radians(-yrot)
    r = math.radians(-zrot)
    sinp, cosp = math.sin(p), math.cos(p)
    sinh, cosh = math.sin(h), math.cos(h)
    sinr, cosr = math.sin(r), math.cos(r)
    sinrsinp = -sinr * -sinp
    cosrsinp = cosr * -sinp

    m = [0.0] * 16
    m[0] = ((cosr * cosh) + (sinrsinp * -sinh)) * xs
    m[4] = (-sinr * cosp) * xs
    m[8] = ((cosr * sinh) + (sinrsinp * cosh)) * xs
    m[12] = x

    m[1] = ((sinr * cosh) + (cosrsinp * -sinh)) * ys
    m[5] = (cosr * cosp) * ys
    m[9] = ((sinr * sinh) + (cosrsinp * cosh)) * ys
    m[13] = y

    m[2] = (cosp * -sinh) * zs
    m[6] = sinp * zs
    m[10] = (cosp * cosh) * zs
    m[14] = z

    m[15] = 1.0
    return m


def create(pos, rot, sca):
    """matrix_create(position, rotation, scale)"""
    return build(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], sca[0], sca[1], sca[2])


def multiply(a, b):
    """matrix_multiply(a, b): primero a, luego b."""
    r = [0.0] * 16
    for i in range(4):
        i4 = i * 4
        a0, a1, a2, a3 = a[i4], a[i4 + 1], a[i4 + 2], a[i4 + 3]
        for j in range(4):
            r[i4 + j] = a0 * b[j] + a1 * b[4 + j] + a2 * b[8 + j] + a3 * b[12 + j]
    return r


def transform_point(m, p):
    x, y, z = p
    return (m[0] * x + m[4] * y + m[8] * z + m[12],
            m[1] * x + m[5] * y + m[9] * z + m[13],
            m[2] * x + m[6] * y + m[10] * z + m[14])


def remove_scale(m):
    m = list(m)
    xs = math.sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2])
    ys = math.sqrt(m[4] * m[4] + m[5] * m[5] + m[6] * m[6])
    zs = math.sqrt(m[8] * m[8] + m[9] * m[9] + m[10] * m[10])
    if xs * ys * zs > 0:
        for i in (0, 1, 2):
            m[i] /= xs
        for i in (4, 5, 6):
            m[i] /= ys
        for i in (8, 9, 10):
            m[i] /= zs
    return m


def remove_rotation(m):
    m = list(m)
    xs = math.sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2])
    ys = math.sqrt(m[4] * m[4] + m[5] * m[5] + m[6] * m[6])
    zs = math.sqrt(m[8] * m[8] + m[9] * m[9] + m[10] * m[10])
    m[0], m[1], m[2] = xs, 0.0, 0.0
    m[4], m[5], m[6] = 0.0, ys, 0.0
    m[8], m[9], m[10] = 0.0, 0.0, zs
    return m


def translate(x, y, z):
    return build(x, y, z, 0, 0, 0, 1, 1, 1)


def rotation(rx, ry, rz):
    return build(0, 0, 0, rx, ry, rz, 1, 1, 1)


def inverse(m):
    """matrix_inverse (4x4 general)."""
    a = [list(m[i * 4:i * 4 + 4]) + [1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    for c in range(4):
        p = max(range(c, 4), key=lambda r: abs(a[r][c]))
        if abs(a[p][c]) < 1e-12:
            return list(IDENTITY)
        a[c], a[p] = a[p], a[c]
        d = a[c][c]
        a[c] = [x / d for x in a[c]]
        for r in range(4):
            if r != c and a[r][c] != 0.0:
                f = a[r][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return [a[i][4 + j] for i in range(4) for j in range(4)]


def position(m):
    return (m[12], m[13], m[14])


# -- vectores (vec3_* / point3D_* de Mine-imator) -------------------------------

def v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def v_len(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def v_normalize(a):
    n = v_len(a)
    return a if n == 0 else (a[0] / n, a[1] / n, a[2] / n)


def v_distance(a, b):
    return v_len(v_sub(a, b))


def v_direction(a, b):
    v = v_normalize(v_sub(b, a))
    return (0.0, 1.0, 0.0) if v_len(v) == 0 else v


def v_normal(vec, angle):
    """vec3_normal(vector, angle en grados)"""
    xx, yy, zz = vec
    ca, sa = math.cos(math.radians(angle)), -math.sin(math.radians(angle))  # lengthdir_x/y
    if abs(zz) != 1:
        cx = ca / math.sqrt(xx * xx + yy * yy + zz * zz)
        cy = sa
        t = (-cx * xx * zz - cy * yy, cy * xx - cx * yy * zz, cx * (xx * xx + yy * yy))
        return v_normalize(t)
    return (ca, sa, 0.0)


def v_rotate_axis_angle(v, axis, angle):
    d = v_mul(axis, v_dot(v, axis))
    r = v_sub(v, d)
    return v_add(d, v_add(v_mul(r, math.cos(angle)), v_mul(v_cross(axis, r), math.sin(angle))))


def v_project_plane(pos, planepos, n):
    return v_sub(pos, v_mul(n, v_dot(n, pos) - v_dot(n, planepos)))


def v_angle_deg(a, b):
    la, lb = v_len(a), v_len(b)
    if la == 0 or lb == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, v_dot(a, b) / (la * lb)))))


def v_angle_signed(a, b, axis):
    d = v_dot(axis, v_cross(a, b))
    s = 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
    return v_angle_deg(a, b) * s


def create_rotate_to(tangent, normal):
    """matrix_create_rotate_to(tangent, normal): marco TBN."""
    b = v_normalize(v_cross(tangent, normal))
    return [b[0], b[1], b[2], 0.0,
            tangent[0], tangent[1], tangent[2], 0.0,
            normal[0], normal[1], normal[2], 0.0,
            0.0, 0.0, 0.0, 1.0]