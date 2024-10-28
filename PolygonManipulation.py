import itertools as itt
import functools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random
import math
from functools import reduce


def draw(polygons, lim_x=100, lim_y=100):
    fig, ax = plt.subplots()

    ax.set_aspect('equal')  # равный масштаб

    ax.set_xlim(-lim_x, lim_x)  # пределы поля
    ax.set_ylim(-lim_y, lim_y)

    ax.spines['left'].set_position('zero')  # перемещение осей + удаление границ
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.annotate('', xy=(lim_x, 0), xytext=(0, 0),  # стрелочки
                arrowprops=dict(arrowstyle='->', color='black'))
    ax.annotate('', xy=(0, lim_y), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black'))

    for tick in ax.xaxis.get_major_ticks():  # настройка цифр
        tick.label1.set_fontsize(10)
        tick.label1.set_color('grey')

    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(10)
        tick.label1.set_color('grey')

    for polygon in polygons:
        poly = Polygon(polygon, closed=True, facecolor='pink', edgecolor='violet', alpha=0.7)
        poly.set_zorder(3)  # слои
        ax.add_patch(poly)

    plt.title('Отрисовка полигонов')
    plt.grid(True)
    plt.show()


# ----------------------------------------------------------------------------- превращения
def tr_translate(coords, distance=10, ly=None): #параллельный перенос
    if ly:
        return abs(distance)
    coords = [(pair[0], pair[1] + distance) for pair in coords]
    return coords



def tr_rotate(coords, angle=45): #поворот
    angle = math.radians(angle)
    coords = [(pair[0] * math.cos(angle) - pair[1] * math.sin(angle),
               pair[0] * math.sin(angle) + pair[1] * math.cos(angle)) for pair in coords]
    return coords



def tr_symmetry(coords, distance=10, ly=None): # симметрия
    if ly:
        return abs(distance)
    axis = distance / 2
    coords = [(pair[0], pair[1] + 2 * (axis - pair[1])) for pair in coords]
    return coords


def tr_homothety(coords, ratios=2, lxy=None): # гомотетия
    if ratios <= 0:
        raise ValueError
    if lxy:
        return ratios
    coords = [(pair[0] * ratios, pair[1] * ratios) for pair in coords]
    return coords


# ----------------------------------------------------------------------------- фильтрация

def flt_convex_polygon(coords): #фильтрации фигур, являющихся выпуклыми многоугольниками
    z = []
    l = len(coords)
    for i in range(l):
        dx1 = coords[(i + 1) % l][0] - coords[(i) % l][0]
        dy1 = coords[(i + 1) % l][1] - coords[(i) % l][1]
        dx2 = coords[(i + 2) % l][0] - coords[(i + 1) % l][0]
        dy2 = coords[(i + 2) % l][1] - coords[(i + 1) % l][1]
        z.append(dx1 * dy2 - dy1 * dx2)
    ans = True if (all([x >= 0 for x in z]) or all([x <= 0 for x in z])) else False

    return ans



def flt_angle_point(coords, dot=[9, 6]): #фильтрации фигур, имеющих хотя бы один угол, совпадающий с заданной точкой
    for i in coords:
        if i == dot:
            return True



def flt_square(coords, s=6): #фильтрации фигур, имеющих площадь меньше заданной
    area = 0
    n = len(coords)
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        area += (x1 * y2 - x2 * y1)

    if abs(area) / 2 < s:
        return True



def flt_short_side(coords, side=3): #фильтрации фигур, имеющих кратчайшую сторону меньше заданного значения
    n = len(coords)
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        if ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 < side:
            return True



def flt_point_inside(polygon, point): #фильтрации выпуклых многоугольников, включающих заданную точку (внутри многоугольника)
    x, y = point
    n = len(polygon)
    inside = False

    if not flt_convex_polygon(polygon):
        raise ValueError

    for i in range(n):
        next_i = (i + 1) % n

        # Проверяем пересечение луча с ребром
        if (polygon[i][1] > y) != (polygon[next_i][1] > y) and \
                x < (polygon[next_i][0] - polygon[i][0]) * (y - polygon[i][1]) / (polygon[next_i][1] - polygon[i][1]) + \
                polygon[i][0]:
            inside = not inside

    return inside




def flt_polygon_angles_inside(coords, polygon): #фильтрации выпуклых многоугольников, включающих любой из углов заданного многоугольника
    if not flt_convex_polygon(coords):
        raise ValueError

    def dop_func(pol):
        angles = []
        n = len(pol)
        for i in range(1, n + 1):
            n_i = (i + 1) % n
            p_i = (i - 1) % n
            i = i % n
            x1, x2, x3 = pol[p_i][0], pol[i][0], pol[n_i][0]
            y1, y2, y3 = pol[p_i][1], pol[i][1], pol[n_i][1]

            vector_ab = (x2 - x1, y2 - y1)
            vector_bc = (x3 - x2, y3 - y2)

            dot_product = vector_ab[0] * vector_bc[0] + vector_ab[1] * vector_bc[1]  # Находим скалярное произведение

            length_ab = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # Находим длины векторов
            length_bc = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)

            cos_theta = dot_product / (length_ab * length_bc)  # Вычисляем угол в радианах
            angle_rad = math.acos(cos_theta)

            angles.append(math.degrees(angle_rad))  # Переводим угол в градусы

        return angles

    coords_angles = dop_func(coords)
    pol_angles = dop_func(polygon)

    for i in coords_angles:
        if i in pol_angles:
            return coords_angles
    return False


# ----------------------------------------------------------------------------- декораторы

def tr_translate_2(distance=9):
    def decorator(func):
        def wrapper(*args, **kwargs):
            p, lx, ly = func(*args, **kwargs)
            d = tr_translate(p, distance, ly=True)  # для предела по у
            mod_pol = tuple(map(lambda x: tr_translate(x, distance), list(p)))
            return mod_pol, lx, ly * 2 + d

        return wrapper

    return decorator


def tr_rotate_2(angle=135):
    def decorator(func):
        def wrapper(*args, **kwargs):
            p, lx, ly = func(*args, **kwargs)
            mod_pol = tuple(map(lambda x: tr_rotate(x, angle), list(p)))
            return mod_pol, lx, lx / 2

        return wrapper

    return decorator


def tr_symmetry_2(distance=9):
    def decorator(func):
        def wrapper(*args, **kwargs):
            p, lx, ly = func(*args, **kwargs)
            d = tr_symmetry(p, distance, ly=True)  # для предела по у
            mod_pol = tuple(map(lambda x: tr_symmetry(x, distance), list(p)))
            return mod_pol, lx, ly * 2 + d

        return wrapper

    return decorator


def tr_homothety_2(ratios=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            p, lx, ly = func(*args, **kwargs)
            d = tr_homothety(p, ratios, lxy=True)  # для предела по у
            mod_pol = tuple(map(lambda x: tr_homothety(x, ratios), list(p)))
            return mod_pol, lx * d, ly * d

        return wrapper

    return decorator


def flt_polygon_angles_inside_2(polygon=[(0, 0), (0, 4), (4, 4), (4, 0)]):
    def decorator(func):
        def wrapper(*args, **kwargs):
            p, lx, ly = func(*args, **kwargs)
            filt_pol = list(filter(lambda x: flt_polygon_angles_inside(x, polygon), p))
            return filt_pol, lx, ly

        return wrapper

    return decorator


# ----------------------------------------------------------------------------- начальные полигоны

@tr_translate_2(20)
def gen_rectangle(ch=None, h=4, w=5, amount=8):
    p_len = -(w * amount + (amount - 1) * (w // 2))  # переменная для установки размеров поля, для прохождения цикла
    if ch:
        p_len = ch * 2 + w  # переделываем начальную координату учитывая коорд прошлой ф + пробел(ее длина)
    pol = (((i, 0), (i, h), (i + w, h), (i + w, 0)) for i in range((p_len // 2), abs(p_len // 2) + 2, w + w // 2))
    if ch:
        return next(pol)
    return pol, 1.5 * p_len // 2, h * 2  # генератор + предел Х + предел У



# @tr_symmetry_2(15)
# @tr_rotate_2()
def gen_triangle(ch=None, side=5, amount=8):
    p_len = -(side * amount + (amount - 1) * (side // 2))
    if ch:
        p_len = ch * 2 + side
    h = side * (3 ** 0.5) / 2

    pol = (((i, 0), (i + (side / 2), h), (i + side, 0)) for i in
           range((p_len // 2), abs(p_len // 2) + 2, side + side // 2))
    if ch:
        return next(pol)
    return pol, 1.5 * p_len // 2, h * 2




def gen_hexagon(ch=None, side=5, amount=8):
    p_len = -(2 * side * amount + (amount - 1) * side)
    h = side * (3 ** 0.5) / 2
    if ch:
        p_len = ch * 2 + 2 * side

    pol = (((i, 0), (i - side / 2, h), (i, h * 2), (i + side, h * 2), (i + 1.5 * side, h), (i + side, 0))
           for i in range((p_len // 2), abs(p_len // 2) + 2, 3 * side))
    if ch:
        return next(pol)
    return pol, 1.5 * p_len // 2, h * 4



# @flt_polygon_angles_inside_2()
def gen_mix(h=5, w=8, side=7):
    d_func = {'r': gen_rectangle, 't': gen_triangle, 'h': gen_hexagon}
    func = ['r', 't', 'h']

    mix = list(itt.product(func, repeat=7))  # список вариаций длины 7 из алфавита func
    some = random.choice(mix)  # берем 1
    while ('r' not in some) or ('t' not in some) or ('h' not in some):  # проверка чтобы входили все фигуры
        some = random.choice(mix)

    rez = []  # общий список коорд
    # формула для высчитывания координат 1 фигуры так, чтобы все фигуры расположились равномерно лево|право
    ch = -((2 * some.count('r') * w + 2 * some.count('t') * side + some.count('h') * 2 * side) // 2)
    for i in some:
        if i == 'r':
            j = d_func[i](ch, h, w, amount=1)  # возвращаем кортеж координат 1 фигуры
        else:
            j = d_func[i](ch, side, amount=1)
        rez.append(j)
        ch = int(j[-1][0]) if i != 'h' else int(
            j[-2][0])  # переделываем начальную координату для след фигуры (берем Х от посл.коорд.)

    return rez, (some.count('r') * w + some.count('t') * side + some.count('h') * 2 * side + 70) // 1.5, (
            side + h) * 1.5
    # генератор + предел Х + предел У




# @tr_homothety_2()
def gen_trapezoid(a=3, b=4, h=5):
    amount = 6
    i = 1
    pol = []
    for j in range(amount):
        k = b / a
        pol.append(((i, -(a / 2)), (i, a / 2), (i + h, b / 2), (i + h, -(b / 2))))
        a = b
        b = a * k
        i += 1.2 * h
    neg_pol = [((-(i[0][0]), i[0][1]), (-(i[1][0]), i[1][1]), (-(i[2][0]), i[2][1]), (-(i[3][0]), i[3][1])) for i in
               pol]

    return neg_pol + pol, i, b



# ----------------------------------------------------------------------------- примеры

# # №1
# p, lx, ly = gen_rectangle(h=5, w = 10, amount = 8)
# d = tr_translate(p, distance = 7,ly=True)  # для предела по у
# pol = list(p)
#
# p_up = list(map(lambda x: tr_translate(x,distance = 7), pol))
# p_down = list(map(lambda x: tr_translate(x,distance = -7), pol))
#
# general = pol + p_up + p_down
# general = tuple(map(lambda x: tr_rotate(x, angle = -30), general))
#
# draw(general, lx, lx )


# # №2
# p, lx, ly = gen_rectangle(h=5, w = 10, amount = 8)
# d = tr_translate(p, distance = 8,ly=True)  # для предела по у
# pol = list(p)
#
# p_up = list(map(lambda x: tr_translate(x,distance = 8), pol))
#
# p_up = list(map(lambda x: tr_rotate(x, angle = 17), p_up))
# pol = list(map(lambda x: tr_rotate(x, angle = -30), pol))
#
# general = p_up + pol
# draw(general, lx, ly * 4 )


# # №3
# p, lx, ly = gen_triangle(side = 6, amount = 7)
# d = tr_symmetry(p, distance = 12,ly=True)  # для предела по у
# pol = list(p)
#
# p_up = list(map(lambda x: tr_symmetry(x,distance = 12), pol))
#
# general = p_up + pol
# draw(general, lx, ly + d)

# # №4
#
# p , lx,ly= gen_trapezoid()
# pol = list(p)
#
# pol = list(map(lambda x: tr_rotate(x, angle = 30), pol))
# draw(pol,lx,lx)


# ----------------------------------------------------------------------------- reduce
pol = [[(1, 0), (0, 4), (4, 4), (7, 0)], [(-3, 4), (-1, 16), (0, 14), (-1, 3)], [(8, 2), (9, 3), (10, 2), (9, 1)]]


def agr_origin_nearest(dot,coords): #поиск угла, самого близкого к началу координат
    n = len(coords)
    x1, y1 = 0, 0
    for i in range(n):
        x2, y2 = dot
        dot_dis = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        x2, y2 = coords[i]
        dot = (x2,y2) if (((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5) < dot_dis else dot

    return dot


def agr_max_side(side,coords): #поиск самого длинной стороны многоугольника
    n = len(coords)
    sides = []
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        sides.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    return max(side,max(sides))



def agr_min_area(s,coords): # поиск самой маленькой площади многоугольника
    area = 0
    n = len(coords)
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        area += (x1 * y2 - x2 * y1)
    area = abs(area) / 2
    return min(s,area)



def agr_perimeter(p,coords): # расчет суммарного периметра
    n = len(coords)
    sides = []
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        sides.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    p+=sum(sides)
    return p


def agr_area(s,coords): #расчет суммарной площади
    area = 0
    n = len(coords)
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        area += (x1 * y2 - x2 * y1)
    s += abs(area) / 2
    return s
