from setuptools import setup, Extension
from Cython.Build import cythonize
from pathlib import Path

PACKAGE_NAME = "src"

def find_extensions(package_name: str):
    base_path = Path(package_name)
    exts = []

    for path in base_path.rglob("*.py"):
        if path.name in [
            "__init__.py",
            "full_attn.py",
            "conv_spconv.py",
            "ray_bg_remover.py",
        ]:
            continue

        # VD: mypkg/core.py  -> module name: mypkg.core
        #     mypkg/sub/algo.py -> module name: mypkg.sub.algo
        module_rel = path.relative_to(base_path).with_suffix("")
        module_name = ".".join((package_name, *module_rel.parts))

        exts.append(
            Extension(
                module_name,
                [str(path)],
            )
        )

    return exts


extensions = find_extensions(PACKAGE_NAME)

setup(
    name=f"{PACKAGE_NAME}_protected",
    version="0.1.0",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",  # Python 3
            # "binding": True,
        },
    ),
    zip_safe=False,
)

# run: python setup.py build_ext --inplace
