"""
Minimal JAX/CUDA cuSolver debug script
Run this in your interactive SLURM session to isolate the issue.
"""

import os
import sys

def test_step(name, func):
    """Run a test step and catch errors."""
    print(f"\n🔍 Testing: {name}")
    try:
        result = func()
        print(f"✅ SUCCESS: {result}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    print("=== JAX cuSolver Debug Script ===")
    
    # Test 1: Basic imports
    test_step("JAX import", lambda: __import__('jax'))
    
    # Test 2: JAX devices
    import jax
    test_step("JAX devices", lambda: jax.devices())
    
    # Test 3: Basic JAX operations
    import jax.numpy as jnp
    test_step("Basic JAX array", lambda: str(jnp.array([1, 2, 3])))
    
    # Test 4: GPU-specific operation
    test_step("GPU array creation", lambda: str(jnp.array([1, 2, 3]).device()))
    
    # Test 5: The problematic operation - QR decomposition
    def test_qr():
        key = jax.random.PRNGKey(0)
        # Small matrix first
        z = jax.random.normal(key, (4, 4))
        q, r = jnp.linalg.qr(z)
        return f"QR shape: {q.shape}, {r.shape}"
    
    qr_success = test_step("Small QR decomposition", test_qr)
    
    # Test 6: Orthogonal initialization (the actual failing code)
    def test_orthogonal():
        from jax import random
        key = jax.random.PRNGKey(42)
        # This is exactly what fails in your code
        orth_matrix = random.orthogonal(key, 256, (), jnp.float32, 256)
        return f"Orthogonal matrix shape: {orth_matrix.shape}"
    
    orth_success = test_step("Orthogonal initialization", test_orthogonal)
    
    # Test 7: Alternative initialization methods
    def test_alternatives():
        key = jax.random.PRNGKey(42)
        # Test alternative that doesn't use cuSolver
        normal_init = jax.random.normal(key, (256, 256))
        return f"Normal init shape: {normal_init.shape}"
    
    test_step("Normal initialization (fallback)", test_alternatives)
    
    # Test 8: Environment variables check
    print(f"\n🔧 Environment Check:")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"JAX_PLATFORM_NAME: {os.environ.get('JAX_PLATFORM_NAME', 'Not set')}")
    
    # Test 9: Force CPU mode as workaround
    def test_cpu_fallback():
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        import jax
        # Restart JAX with CPU
        jax.config.update('jax_platform_name', 'cpu')
        from jax import random
        key = jax.random.PRNGKey(42)
        orth_matrix = random.orthogonal(key, 256, (), jnp.float32, 256)
        return f"CPU orthogonal shape: {orth_matrix.shape}"
    
    print(f"\n🔧 Testing CPU fallback...")
    cpu_success = test_step("CPU orthogonal initialization", test_cpu_fallback)
    
    # Summary and recommendations
    print(f"\n📊 SUMMARY:")
    if orth_success:
        print("✅ GPU cuSolver is working - the error might be intermittent")
    elif cpu_success:
        print("⚠️  GPU cuSolver broken, but CPU works")
        print("💡 WORKAROUND: Add this to your main.py before importing JAX:")
        print("   import os; os.environ['JAX_PLATFORM_NAME'] = 'cpu'")
    else:
        print("❌ Both GPU and CPU failed - deeper JAX installation issue")
        
    print(f"\n🔧 QUICK FIXES TO TRY:")
    print("1. Restart interactive session")
    print("2. Set: export JAX_PLATFORM_NAME=cpu")
    print("3. Use different initialization: jax.nn.initializers.normal() instead of orthogonal")

if __name__ == "__main__":
    main()