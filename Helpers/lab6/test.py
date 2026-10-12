import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

N = 10

a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)

c = np.empty_like(a)

a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

mod = SourceModule("""
    __global__ void add_vectors(float *a, float *b, float *c){
        int idx = threadIdx.x;
        c[idx] = a[idx] + b[idx];               
    }
""")

# Получаем функцию `add_vectors` из CUDA модуля
add_vectors = mod.get_function("add_vectors")


# Запускаем kernel с количеством потоков, равным длине массива
add_vectors(a_gpu, b_gpu, c_gpu, block=(N,1,1))

# Копируем результат из GPU памяти в CPU память
cuda.memcpy_dtoh(c, c_gpu)

# Выводим результат
print("a = ", a)
print("b = ", b)

print("c = a + b", c)
