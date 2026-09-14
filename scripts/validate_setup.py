#!/usr/bin/env python3
"""
Environment and setup validation script.

Run this to verify that the development environment is correctly
configured before launching the application.

Usage:
    python scripts/validate_setup.py
"""

import sys
import os
import importlib

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_python_version():
    """Verify Python version is 3.10+."""
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 10:
        print(f"  [OK] Python {major}.{minor}")
        return True
    else:
        print(f"  [FAIL] Python {major}.{minor} - need 3.10+")
        return False


def check_import(module_name, min_version=None):
    """Verify a module can be imported and meets minimum version."""
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  [OK] {module_name} {version}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {module_name} - {e}")
        return False


def check_mediapipe_tasks_api():
    """Verify MediaPipe Tasks API is available."""
    try:
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
        from mediapipe.tasks.python import BaseOptions
        print("  [OK] MediaPipe Tasks API (HandLandmarker)")
        return True
    except ImportError as e:
        print(f"  [FAIL] MediaPipe Tasks API - {e}")
        return False


def check_panda3d_core():
    """Verify Panda3D core imports work."""
    try:
        from panda3d.core import Point3, Vec3, NodePath
        from panda3d.core import GeomVertexFormat, GeomVertexData
        print("  [OK] Panda3D core (Point3, Vec3, GeomVertex)")
        return True
    except ImportError as e:
        print(f"  [FAIL] Panda3D core - {e}")
        return False


def check_numpy_linalg():
    """Verify numpy.linalg.eigh works."""
    try:
        import numpy as np
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        vals, vecs = np.linalg.eigh(A)
        assert len(vals) == 2
        print("  [OK] NumPy eigh (symmetric eigendecomposition)")
        return True
    except Exception as e:
        print(f"  [FAIL] NumPy eigh - {e}")
        return False


def check_project_config():
    """Verify project config is importable."""
    try:
        from src.config import AppConfig, DEFAULT_CONFIG
        assert DEFAULT_CONFIG.data.seed == 42
        assert DEFAULT_CONFIG.data.n_points > 0
        print(f"  [OK] Project config (seed={DEFAULT_CONFIG.data.seed}, N={DEFAULT_CONFIG.data.n_points})")
        return True
    except Exception as e:
        print(f"  [FAIL] Project config - {e}")
        return False


def main():
    print("=" * 60)
    print("Environment & Setup Validation")
    print("=" * 60)

    all_ok = True

    print("\n[1] Python Version")
    all_ok &= check_python_version()

    print("\n[2] Core Dependencies")
    for mod in ['numpy', 'scipy', 'cv2', 'sklearn', 'matplotlib', 'PIL']:
        all_ok &= check_import(mod)

    print("\n[3] Project-Specific Dependencies")
    all_ok &= check_import('mediapipe')
    all_ok &= check_import('panda3d')
    all_ok &= check_import('pytest')

    print("\n[4] Deep API Checks")
    all_ok &= check_mediapipe_tasks_api()
    all_ok &= check_panda3d_core()
    all_ok &= check_numpy_linalg()

    print("\n[5] Project Structure")
    all_ok &= check_project_config()

    print("\n" + "=" * 60)
    if all_ok:
        print("[SUCCESS] ALL CHECKS PASSED - Environment is fully operational.")
    else:
        print("[FAIL] SOME CHECKS FAILED - Fix issues above before proceeding.")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
