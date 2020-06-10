
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name="_warpctc",
    ext_modules=[
        CUDAExtension("_warpctc", 
        	sources=["src/binding.cpp", "src/reduce.cu", "src/ctc_entrypoint.cu"],
        	include_dirs=["include"],
        	extra_compile_args=['-DWARPCTC_ENABLE_GPU', '-D__CUDACC_X__'])
    ], 
    cmdclass={
        "build_ext": BuildExtension
    }
)