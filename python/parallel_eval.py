import phasefield as material
import time
from multiprocessing import Queue
from multiprocessing import Process as Worker

def parallel_eval(n, fun):
    tasks = []
    qs = []

    for i in range(0, n):
        q = Queue()
        t = Worker(target=fun, args=(i, q))
        tasks.append(t)
        qs.append(q)
        t.start()
        
    for t in tasks:
        t.join()

    expr = []

    for q in qs:
        expr.append(q.get())

    return expr

if __name__ == '__main__':
    tick = time.time()
    ntf = material.n_test_functions()

    print("--------------------------------")

    energy_expr = material.makeenergy()
    material.c_code(energy_expr)

    print("--------------------------------")

    grad_expr = parallel_eval(ntf, material.makegrad)
    material.c_code(grad_expr)

    hessian_tuples = parallel_eval(ntf, material.makehessian)

    hessian_expr = [0]*(ntf*ntf)
    
    for hts in hessian_tuples:
        for ht in hts:
            i, j, expression = ht
            hessian_expr[i*ntf + j] = expression

    material.c_code(hessian_expr)

    tock = time.time()
    print(f'Code generation took {round(tock - tick, 4)} seconds')
