import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def nearest_neighbour_rule(rule):
    n = rule.shape[0]
    
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(rule[i, :3] - rule[j, :3])
            
    coord = np.zeros((3))
    print(f"coord: {coord}")
    
    new_rule = []
    used = []
    
    # put n growing number in a list
    todo = []
    for i in range(n):
        todo.append(i)
        
    # search the nearest neighbour for coord
    while len(todo) > 0:
        min_dist = np.inf
        for i in todo:
            dist = np.linalg.norm(rule[i, :3] - coord)
            if dist < min_dist:
                min_dist = dist
                index = i
                
        coord = rule[index, :3]
        # remove index from todo
        todo.remove(index)
        
        new_rule.append(rule[index, :])
        
    return np.array(new_rule)
        

def generate_rule(file_name, precision=16):
    """
    Generate a rule from a csv file.
    It is assumed that the csv file has the following format:
    | a_{i,0} | a_{i,1} | a_{i,2} | a_{i,3} | w_i |
    where a_{i,j} is the j-th element of the i-th rule and w_i is the weight of the i-th rule.
    
    Cite the following paper for more information: Symmetric quadrature rules for tetrahedra based on a cubic close-packed lattice arrangement Lee Shunn, Frank Ham
    """
    rule_base = pd.read_csv(file_name, header=None).values
    
    rule = np.zeros((rule_base.shape[0], 4), dtype=np.float64)
    
    thet_e = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    for row_i in range(rule_base.shape[0]):
        s = np.zeros(3)
        for i in range(4):
            rule[row_i, :3] = rule[row_i, :3] + thet_e[i, :] * rule_base[row_i, i]
        
        rule[row_i, 3] = rule_base[row_i, 4]
    
    rule = nearest_neighbour_rule(rule)
    
    return rule


def relative_error(a, b):
    """
    Calculate the relative error between two numbers.
    """
    return np.abs(a - b) / np.abs(a)

def rule_to_c(rule):
    """
    Convert a rule to a c++ string.
    """
    TET4_NQP = rule.shape[0]
    
    def column_to_nr_str(column, precision):
        """
        Convert a column to a string with precision.
        """
        rule_str = ''
        for i in range(rule.shape[0]):
            comma = ', ' if i < rule.shape[0] - 1 else ''
            rule_str += f"{rule[i, column]:.{precision}f}{comma}"
        return rule_str
            
        
    
    rule_str = f"#define TET4_NQP  {TET4_NQP}\n"
    rule_str += "static real_t tet4_qw[TET4_NQP] = {\n"
    rule_str += column_to_nr_str(3, 16)
    rule_str += "};\n\n"
    
    rule_str += "static real_t tet4_qx[TET4_NQP] = {\n"
    rule_str += column_to_nr_str(0, 16)
    rule_str += "};\n\n"
    
    rule_str += "static real_t tet4_qy[TET4_NQP] = {\n"
    rule_str += column_to_nr_str(1, 16)
    rule_str += "};\n\n"
    
    rule_str += "static real_t tet4_qz[TET4_NQP] = {\n"
    rule_str += column_to_nr_str(2, 16)
    rule_str += "};\n\n"
    
    
    return rule_str

def main_rule():
    """
        Returns the main rule for the tet4 element.

    Returns:
        np.array: the main rule for the tet4 element.
        it is a 56 point rule the first three columns are the x, y, z coordinates and the fourth column is the weight.
    """
    tet4_qx = [0.2500000000000000, 0.6175871903000830, 0.1274709365666390, 0.1274709365666390,
            0.1274709365666390, 0.9037635088221031, 0.0320788303926323, 0.0320788303926323,
            0.0320788303926323, 0.4502229043567190, 0.0497770956432810, 0.0497770956432810,
            0.0497770956432810, 0.4502229043567190, 0.4502229043567190, 0.3162695526014501,
            0.1837304473985499, 0.1837304473985499, 0.1837304473985499, 0.3162695526014501,
            0.3162695526014501, 0.0229177878448171, 0.2319010893971509, 0.2319010893971509,
            0.5132800333608811, 0.2319010893971509, 0.2319010893971509, 0.2319010893971509,
            0.0229177878448171, 0.5132800333608811, 0.2319010893971509, 0.0229177878448171,
            0.5132800333608811, 0.7303134278075384, 0.0379700484718286, 0.0379700484718286,
            0.1937464752488044, 0.0379700484718286, 0.0379700484718286, 0.0379700484718286,
            0.7303134278075384, 0.1937464752488044, 0.0379700484718286, 0.7303134278075384,
            0.1937464752488044]

    tet4_qy = [0.2500000000000000, 0.1274709365666390, 0.1274709365666390, 0.1274709365666390,
            0.6175871903000830, 0.0320788303926323, 0.0320788303926323, 0.0320788303926323,
            0.9037635088221031, 0.0497770956432810, 0.4502229043567190, 0.0497770956432810,
            0.4502229043567190, 0.0497770956432810, 0.4502229043567190, 0.1837304473985499,
            0.3162695526014501, 0.1837304473985499, 0.3162695526014501, 0.1837304473985499,
            0.3162695526014501, 0.2319010893971509, 0.0229177878448171, 0.2319010893971509,
            0.2319010893971509, 0.5132800333608811, 0.2319010893971509, 0.0229177878448171,
            0.5132800333608811, 0.2319010893971509, 0.5132800333608811, 0.2319010893971509,
            0.0229177878448171, 0.0379700484718286, 0.7303134278075384, 0.0379700484718286,
            0.0379700484718286, 0.1937464752488044, 0.0379700484718286, 0.7303134278075384,
            0.1937464752488044, 0.0379700484718286, 0.1937464752488044, 0.0379700484718286,
            0.7303134278075384]
    
    tet4_qz = [0.2500000000000000, 0.1274709365666390, 0.1274709365666390, 0.6175871903000830,
            0.1274709365666390, 0.0320788303926323, 0.0320788303926323, 0.9037635088221031,
            0.0320788303926323, 0.0497770956432810, 0.0497770956432810, 0.4502229043567190,
            0.4502229043567190, 0.4502229043567190, 0.0497770956432810, 0.1837304473985499,
            0.1837304473985499, 0.3162695526014501, 0.3162695526014501, 0.3162695526014501,
            0.1837304473985499, 0.2319010893971509, 0.2319010893971509, 0.0229177878448171,
            0.2319010893971509, 0.2319010893971509, 0.5132800333608811, 0.5132800333608811,
            0.2319010893971509, 0.0229177878448171, 0.0229177878448171, 0.5132800333608811,
            0.2319010893971509, 0.0379700484718286, 0.0379700484718286, 0.7303134278075384,
            0.0379700484718286, 0.0379700484718286, 0.1937464752488044, 0.1937464752488044,
            0.0379700484718286, 0.7303134278075384, 0.7303134278075384, 0.1937464752488044,
            0.0379700484718286]
    
    tet4_qw = [-0.2359620398477559, 0.0244878963560563, 0.0244878963560563, 0.0244878963560563,
            0.0244878963560563,  0.0039485206398261, 0.0039485206398261, 0.0039485206398261,
            0.0039485206398261,  0.0263055529507371, 0.0263055529507371, 0.0263055529507371,
            0.0263055529507371,  0.0263055529507371, 0.0263055529507371, 0.0829803830550590,
            0.0829803830550590,  0.0829803830550590, 0.0829803830550590, 0.0829803830550590,
            0.0829803830550590,  0.0254426245481024, 0.0254426245481024, 0.0254426245481024,
            0.0254426245481024,  0.0254426245481024, 0.0254426245481024, 0.0254426245481024,
            0.0254426245481024,  0.0254426245481024, 0.0254426245481024, 0.0254426245481024,
            0.0254426245481024,  0.0134324384376852, 0.0134324384376852, 0.0134324384376852,
            0.0134324384376852,  0.0134324384376852, 0.0134324384376852, 0.0134324384376852,
            0.0134324384376852,  0.0134324384376852, 0.0134324384376852, 0.0134324384376852,
            0.0134324384376852]

    # Convert to numpy arrays
    tet4_qx_np = np.array(tet4_qx)
    tet4_qy_np = np.array(tet4_qy)
    tet4_qz_np = np.array(tet4_qz)
    tet4_qw_np = np.array(tet4_qw)
    
    rule = np.array([tet4_qx_np, tet4_qy_np, tet4_qz_np, tet4_qw_np]).T
    return rule


def test_rule(rule, f):
    """
    Test a rule by integrating a function f.
    """
    w_tot = np.sum(rule[:, 3])
    
    integral = 0.0
    for i in range(rule.shape[0]):
        integral += rule[i, 3] * f(rule[i, 0], rule[i, 1], rule[i, 2])
    
    integral = 1.0 / 6.0 * integral / w_tot
    
    return integral

def test_quad_montecarlo(f, n=10000):
    """
    Test a rule by integrating a function f using the monte carlo method.
    """
    w_tot = np.sum(rule[:, 3])
    
    integral = 0.0
    cnt = 0
    while cnt < n:
        r = np.random.rand(3)
        # print(cnt, r, f(r[0], r[1], r[2]))
        if r[0] <= 1.0 and r[1] <= (1.0 - r[0]) and r[2] <= (1.0 - r[0] - r[1]):
            integral += f(r[0], r[1], r[2])
            cnt += 1
        else:
            continue
    
    integral = 1.0 / (6.0 * cnt) * integral
    
    return integral
     
if __name__ == '__main__':
    
    rule_nr = 56
    # rule_nr = 35
    
    rule = generate_rule(f'rule{rule_nr}.csv')
    
    print(rule)
    
    rule_c = rule_to_c(rule)
    # print(rule_c)
    
    with open(f'rule{rule_nr}.h', 'w') as f:
        f.write(rule_c)
    
    w_tot = np.sum(rule[:, 3])
    
    # make a scatter plot of the rule
    # where the first three columns are the x, y, z coordinates and the fourth column is the weight
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(rule[:, 0], rule[:, 1], rule[:, 2], s=rule[:, 3]*20000)
    
    print(f"w_tot: {w_tot}")
    
    fun_list = []
    fun_list.append(lambda x, y, z: np.sin(x+y+z) + np.cos(x+y+z))
    fun_list.append(lambda x, y, z: np.sin(x+y+z)**2 + np.cos(x+y**2+z))
    fun_list.append(lambda x, y, z: np.sin(x+y+z) + np.exp(x+y**2+z))
    fun_list.append(lambda x, y, z: np.exp((x + y + z)**2) + np.exp(x + y**2 + z))
    fun_list.append(lambda x, y, z: np.sin(2*x + 2*y + 4*z) + np.cos(2*x + 4*y + 2*z))
    fun_list.append(lambda x, y, z: np.sin(2*x + 2*y + 4*z)**2 + np.cos(2*x + 4*y + 2*z)**2)
    fun_list.append(lambda x, y, z: np.sin(5*x + 6*y + 4*z)**2 + np.cos(2*x + y + 9*z)**2)
    
    
    fun_solutions = [0.2311889512061578294277, 
                     0.2136218568024591402653, 
                     0.4205361750387262663085, 
                     0.6228215579347258075946, 
                     0.06723507722240459814545,
                     0.1666666666666666666667,
                     0.1680123009033446890659,
                     ]
    
    for i, f in enumerate(fun_list):
        print(f"integral_{i}: {test_rule(rule, f)}")
        print(f"exact_{i}:    {fun_solutions[i]}")
        print(f"error_{i}:    {relative_error(test_rule(rule, f), fun_solutions[i])}")
        print()
        
    rule_original = main_rule()
    
    print("--------------------------")
    print("Original rule")
    for i, f in enumerate(fun_list):
        print(f"integral_{i}: {test_rule(rule_original, f)}")
        print(f"exact_{i}:    {fun_solutions[i]}")
        print(f"error_{i}:    {relative_error(test_rule(rule_original, f), fun_solutions[i])}")
        print()
    
    # fn = 6
    # r = test_quad_montecarlo(fun_list[fn], 50000)
    # print(f"montecarlo: {r}, error: {np.abs(r - fun_solutions[fn])}")
    
    # tests for montecarlo
    samples = 200000
    print()
    print("--------------------------")
    print(f"Montecarlo tests, samples: {samples}")
    for i, f in enumerate(fun_list):
        r = test_quad_montecarlo(f, samples)
        print(f"montecarlo_{i}: {r}, \t error: {relative_error(r, fun_solutions[i])}")
    
    plt.show()
    
