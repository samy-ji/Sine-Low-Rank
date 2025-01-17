# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# setup(
#     name='matmul_cuda',
#     ext_modules=[
#         CUDAExtension('matmul_cuda', [
#             'matmul_wrapper.cpp',
#             'matrixMul.cu',
#         ]),
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     }
# )

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# setup(
#     name='cuda_matmul',
#     ext_modules=[
#         CUDAExtension('cuda_matmul', [
#             'matmul_kernel.cu',
#         ]),
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     })

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_matmul',
    ext_modules=[
        CUDAExtension('cuda_matmul', [
            'matmul_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })