# SPDX-License-Identifier: MIT
try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError as _e:
    # pytest 会以裸模块身份导入包根(无父包), 相对导入失效; ComfyUI 运行时
    # 以正规包身份加载, 走不到这里。兜底: 以包身份重载自身后取注册表。
    if 'attempted relative import' not in str(_e):
        raise
    import importlib.util as _ilu
    import os as _os
    import sys as _sys
    _dir = _os.path.dirname(_os.path.abspath(__file__))
    _name = _os.path.basename(_dir)
    _spec = _ilu.spec_from_file_location(_name, _os.path.join(_dir, '__init__.py'),
                                         submodule_search_locations=[_dir])
    _pkg = _ilu.module_from_spec(_spec)
    _sys.modules[_name] = _pkg
    _spec.loader.exec_module(_pkg)
    NODE_CLASS_MAPPINGS = _pkg.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = _pkg.NODE_DISPLAY_NAME_MAPPINGS

__version__ = "1.0.0"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
