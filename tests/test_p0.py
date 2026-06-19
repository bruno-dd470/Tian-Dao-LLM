"""
Tests de validation des corrections P0 (bugs critiques).
Usage :
    pytest tests/test_p0.py -v
"""
import threading
from Endoregulated_AI_v27 import EndoRegulatedCore, get_core_lock


# ============================================================================
# TEST P0-A : Le flip persiste (bug encode_bits corrigé)
# ============================================================================
def test_flip_persists():
    """Vérifie que encode_bits modifie VRAIMENT l'état du système."""
    core = EndoRegulatedCore(seed=42)
    
    initial_mode_F = core.attractor_mode['F']
    core.encode_bits(5)
    mode_after_1 = core.attractor_mode['F']
    core.encode_bits(5)
    mode_after_2 = core.attractor_mode['F']
    
    assert mode_after_1 != initial_mode_F or mode_after_2 != mode_after_1, \
        "❌ P0-A : Le flip est annulé ! encode_bits ne propage pas le changement."
    
    print(f"✅ P0-A : flip persiste (F: {initial_mode_F} → {mode_after_1} → {mode_after_2})")


# ============================================================================
# TEST P0-B : Thread safety
# ============================================================================
def test_thread_safety():
    """Vérifie que plusieurs threads peuvent utiliser le core sans collision."""
    errors = []
    
    def worker(thread_id):
        try:
            c = EndoRegulatedCore(seed=thread_id)
            for _ in range(50):
                c.encode_bits(7)
                c.step()
        except Exception as e:
            errors.append((thread_id, e))
    
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert not errors, f"❌ P0-B : Erreurs thread : {errors}"
    print("✅ P0-B : thread safety OK (10 threads × 50 itérations)")


# ============================================================================
# TEST P0-C : Déterminisme
# ============================================================================
def test_determinism():
    """Vérifie que deux cores avec le même seed produisent le même état."""
    core1 = EndoRegulatedCore(seed=42)
    core2 = EndoRegulatedCore(seed=42)
    
    core1.encode_bits(10)
    core2.encode_bits(10)
    
    assert core1.pentad_sign == core2.pentad_sign, \
        "❌ P0-C : Deux cores avec seed=42 divergent !"
    assert core1.attractor_mode == core2.attractor_mode, \
        "❌ P0-C : Les modes des attracteurs divergent !"
    
    print("✅ P0-C : déterminisme OK (seed=42 reproductible)")


# ============================================================================
# TEST P0-D : Lock global accessible
# ============================================================================
def test_lock_available():
    """Vérifie que le lock de protection est bien exporté."""
    lock = get_core_lock()
    assert lock is not None, "❌ P0-D : get_core_lock() retourne None"
    assert isinstance(lock, type(threading.Lock())), \
        "❌ P0-D : get_core_lock() ne retourne pas un Lock"
    print("✅ P0-D : lock global accessible")


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 VALIDATION DES CORRECTIONS P0")
    print("=" * 60)
    
    test_flip_persists()
    test_thread_safety()
    test_determinism()
    test_lock_available()
    
    print("\n" + "=" * 60)
    print("🎉 TOUS LES TESTS P0 PASSENT !")
    print("=" * 60)
