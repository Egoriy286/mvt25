import numpy as np
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Параметры изображения
width, height = 3840, 2160
max_iter = 100

# Границы на комплексн  ой плоскости
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5

# Преобразуем в массивы numpy для передачи на GPU
image = np.zeros((height, width), dtype=np.float32)

# CUDA-код для вычисления множества Мандельброта
mandelbrot_kernel = SourceModule("""
__global__ void mandelbrot(float *image, int width, int height, 
                           float xmin, float xmax, float ymin, float ymax, int max_iter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float real = xmin + x * (xmax - xmin) / width;
    float imag = ymin + y * (ymax - ymin) / height;
    float c_real = real;
    float c_imag = imag;

    int value = 0;
    for (int i = 0; i < max_iter; i++) {
        float real2 = real * real;
        float imag2 = imag * imag;
        if (real2 + imag2 > 4.0f) {
            value = i;
            break;
        }
        imag = 2 * real * imag + c_imag;
        real = real2 - imag2 + c_real;
    }
    image[y * width + x] = (float)value;
}
""")

# Получаем функцию из модуля
mandelbrot = mandelbrot_kernel.get_function("mandelbrot")

# Выделяем память на GPU для изображения
image_gpu = cuda.mem_alloc(image.nbytes)

# Задаём параметры сетки и блоков
block_dim = (16, 16, 1)  # Блоки 16x16
grid_dim = ((width + block_dim[0] - 1) // block_dim[0],
            (height + block_dim[1] - 1) // block_dim[1])

# Запускаем вычисление на GPU
mandelbrot(image_gpu, np.int32(width), np.int32(height),
           np.float32(xmin), np.float32(xmax), np.float32(ymin), np.float32(ymax), 
           np.int32(max_iter), block=block_dim, grid=grid_dim)

# Копируем результат с GPU на CPU
cuda.memcpy_dtoh(image, image_gpu)

# Отображение изображения
plt.imshow(image, extent=(xmin, xmax, ymin, ymax), cmap="hot")
plt.colorbar(label="Итерации до выхода за пределы")
plt.title("Множество Мандельброта с использованием PyCUDA")
plt.xlabel("Re(c)")
plt.ylabel("Im(c)")
plt.show()
