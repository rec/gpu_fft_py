from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension

setup(
  cmdclass = {'build_ext': build_ext},
  ext_modules = [
    Extension("gpu_fft_py",
              sources=["gpu_fft_py.pyx",
                       "gpu_fft/gpu_fft.c",
                       "gpu_fft/gpu_fft_shaders.c",
                       "gpu_fft/gpu_fft_twiddles.c",
                       "gpu_fft/mailbox.c",
                       ],
              extra_compile_args=["-Igpu_fft"],
              extra_link_args=["-lrt", "-lm"],
              )
    ]
)


#  ext_modules = cythonize('gpu_fft.pyx',
#                          extra_link_args='-Lgpu_fft -lgpufft')
