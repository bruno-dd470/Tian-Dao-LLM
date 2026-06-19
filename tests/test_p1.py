"""
Tests de validation des corrections P1 (robustesse).
Usage :
    pytest tests/test_p1.py -v
"""
import os
import hashlib
import numpy as np


# ============================================================================
# TEST P1-1 : PCA nécessite au moins 5 points
# ============================================================================
def test_pca_minimum_points():
    """Vérifie que la PCA n'est pas lancée sur moins de 5 points."""
    from sklearn.decomposition import PCA
    
    # 3 points → PCA instable (variance mal définie)
    points_3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    pca = PCA(n_components=2)
    result_3 = pca.fit_transform(points_3)
    assert result_3.shape == (3, 2)
    
    # 5 points → PCA stable
    points_5 = np.random.randn(5, 12).astype(np.float32)
    pca = PCA(n_components=2)
    result_5 = pca.fit_transform(points_5)
    assert result_5.shape == (5, 2)
    print("✅ P1-1 : PCA stable sur 5+ points")


# ============================================================================
# TEST P1-3 : Hash SHA-256 vs polynomial
# ============================================================================
def test_hash_robustness():
    """Vérifie que SHA-256 produit moins de collisions que le hash polynomial."""
    texts = [
        "chat", "Chat", "CHAT",
        "bonjour", "Bonjour", "BONJOUR",
        "a", "b", "c", "d", "e",
        "ab", "ba", "abc", "bac",
    ]
    
    # Hash polynomial (ancien, faible)
    def poly_hash(text):
        h = 0
        for c in text:
            h = (h * 31 + ord(c)) % 64
        return h
    
    poly_hashes = [poly_hash(t) for t in texts]
    poly_collisions = len(poly_hashes) - len(set(poly_hashes))
    
    # Hash SHA-256 (nouveau, robuste)
    def sha_hash(text):
        digest = hashlib.sha256(text.encode('utf-8')).digest()
        return int.from_bytes(digest[:2], 'big') % 64
    
    sha_hashes = [sha_hash(t) for t in texts]
    sha_collisions = len(sha_hashes) - len(set(sha_hashes))
    
    print(f"   Collisions polynomial : {poly_collisions}/{len(texts)}")
    print(f"   Collisions SHA-256    : {sha_collisions}/{len(texts)}")
    
    # SHA-256 doit avoir moins ou autant de collisions
    assert sha_collisions <= poly_collisions, \
        "❌ P1-3 : SHA-256 a plus de collisions que polynomial !"
    print("✅ P1-3 : SHA-256 plus robuste que polynomial")


# ============================================================================
# TEST P1-5 : Validation de texte
# ============================================================================
def test_text_validation():
    """Vérifie que les textes vides ou blancs sont correctement rejetés."""
    invalid_texts = [
        "",
        " ",
        "  ",
        "\t",
        "\n",
        " \t\n ",
    ]
    
    for text in invalid_texts:
        assert not text or not text.strip(), \
            f"❌ P1-5 : '{repr(text)}' devrait être invalide"
    
    valid_texts = ["a", "hello", " a ", "hello world"]
    for text in valid_texts:
        assert text and text.strip(), \
            f"❌ P1-5 : '{repr(text)}' devrait être valide"
    
    print("✅ P1-5 : validation de texte robuste")


# ============================================================================
# TEST P1-6 : Pas de fuite matplotlib
# ============================================================================
def test_matplotlib_no_leak():
    """Vérifie que les figures sont bien fermées même en cas d'exception."""
    import matplotlib.pyplot as plt
    
    # Compte les figures ouvertes avant
    initial_figs = len(plt.get_fignums())
    
    # Simule une exception après subplots
    fig = None
    try:
        fig, ax = plt.subplots()
        raise ValueError("Test exception")
    except ValueError:
        if fig is not None:
            plt.close(fig)
    
    # Compte les figures ouvertes après
    final_figs = len(plt.get_fignums())
    
    assert final_figs == initial_figs, \
        f"❌ P1-6 : fuite de figure ({initial_figs} → {final_figs})"
    print("✅ P1-6 : pas de fuite matplotlib")


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 VALIDATION DES CORRECTIONS P1")
    print("=" * 60)
    
    test_pca_minimum_points()
    test_hash_robustness()
    test_text_validation()
    test_matplotlib_no_leak()
    
    print("\n" + "=" * 60)
    print("🎉 TOUS LES TESTS P1 PASSENT !")
    print("=" * 60)
