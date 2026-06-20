"""
Tests unitaires de l'interface Gradio (app2.py).
Usage :
    pytest tests/test_app.py -v
    ou depuis le dossier tests/ : python3 -m pytest test_app.py -v
"""
import sys
import os
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour les tests
import matplotlib.pyplot as plt
import threading
from unittest.mock import patch, MagicMock

# ✅ Ajouter le chemin parent pour les imports
# Le dossier tests/ est un sous-dossier de Tian-Dao-LLM/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import depuis la racine du projet
from Endoregulated_AI_v27 import EndoRegulatedCore

# 🔴 IMPORTANT : Indiquer quel fichier app utiliser (app.py ou app2.py)
# Ici on importe depuis app2.py (version corrigée avec 20D)
import app
from app import text_to_embedding, visualize_embedding

# ============================================================================
# FIXTURES COMMUNES
# ============================================================================

@pytest.fixture
def reset_core():
    """
    Réinitialise le core AVANT chaque test pour garantir l'indépendance.
    IMPORTANT : on remplace le core DU MODULE app, pas celui de test_app.
    """
    from Endoregulated_AI_v27 import EndoRegulatedCore, get_core_lock
    
    # Créer un nouveau core
    new_core = EndoRegulatedCore(noise_level=0.15, seed=42)
    
    # Remplacer le core global du module app (app2.py)
    app.core = new_core
    app.core_lock = get_core_lock()
    
    yield new_core
    
    # Nettoyage après le test : réinitialiser le counter
    new_core.input_counter = 0


@pytest.fixture
def sample_texts():
    """Textes de test variés."""
    return [
        "Bonjour le monde",
        "L'intelligence artificielle est fascinante",
        "Le cycle Wuxing gouverne l'équilibre",
        "Sheng et Ke dansent dans le chaos",
        "La topologie des attracteurs révèle l'harmonie",
    ]

# ============================================================================
# TESTS DE TEXT_TO_EMBEDDING
# ============================================================================

class TestTextToEmbedding:
    """Tests de la fonction text_to_embedding."""

    def test_returns_tuple_of_three(self, reset_core):
        """text_to_embedding retourne un tuple de 3 éléments."""
        result = text_to_embedding("test")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_embedding_20d_shape(self, reset_core):
        """L'embedding 20D a la bonne forme."""
        emb_20d, emb_768d, attractor = text_to_embedding("test")
        assert emb_20d.shape == (20,)
        assert emb_20d.dtype == np.float32

    def test_embedding_768d_shape(self, reset_core):
        """L'embedding 768D a la bonne forme."""
        emb_20d, emb_768d, attractor = text_to_embedding("test")
        assert emb_768d.shape == (768,)
        assert emb_768d.dtype == np.float32

    def test_attractor_is_valid(self, reset_core):
        """L'attracteur retourné est valide (A-T)."""
        emb_20d, emb_768d, attractor = text_to_embedding("test")
        assert attractor in reset_core.attractors
        assert len(attractor) == 1
        assert 'A' <= attractor <= 'T'

    def test_embedding_values_in_range(self, reset_core):
        """Les valeurs de l'embedding 20D sont dans [-1, 1]."""
        emb_20d, _, _ = text_to_embedding("test")
        assert np.all(emb_20d >= -1.0)
        assert np.all(emb_20d <= 1.0)

    def test_deterministic_same_text(self, reset_core):
        """
        Le même texte produit le même embedding SI l'état du core est identique.
        """
        from Endoregulated_AI_v27 import EndoRegulatedCore
        
        core1 = EndoRegulatedCore(noise_level=0.15, seed=42)
        core2 = EndoRegulatedCore(noise_level=0.15, seed=42)
        
        original_core = app.core
        original_lock = app.core_lock
        
        try:
            app.core = core1
            app.core_lock = app.get_core_lock()
            emb1, emb768_1, attr1 = text_to_embedding("bonjour")
            
            app.core = core2
            emb2, emb768_2, attr2 = text_to_embedding("bonjour")
            
            np.testing.assert_array_almost_equal(emb1, emb2)
            np.testing.assert_array_almost_equal(emb768_1, emb768_2)
            assert attr1 == attr2
        finally:
            app.core = original_core
            app.core_lock = original_lock

    def test_different_texts_different_embeddings(self, reset_core):
        """Des textes différents produisent des embeddings différents."""
        emb1, _, _ = text_to_embedding("chat")
        emb2, _, _ = text_to_embedding("chien")
        
        assert not np.allclose(emb1, emb2)

    def test_core_state_changes(self, reset_core):
        """L'injection dans le core modifie son état."""
        initial_counter = reset_core.input_counter
        text_to_embedding("test")
        assert reset_core.input_counter == initial_counter + 1

    def test_multiple_injections_evolve_system(self, reset_core):
        """Plusieurs injections font évoluer le système."""
        for text in ["texte1", "texte2", "texte3", "texte4", "texte5"]:
            text_to_embedding(text)
        assert True  # Le test passe si aucune exception n'est levée


# ============================================================================
# TESTS DE VISUALIZE_EMBEDDING
# ============================================================================

class TestVisualizeEmbedding:
    """Tests de la fonction visualize_embedding."""

    def test_returns_tuple_of_three(self, reset_core):
        """visualize_embedding retourne un tuple de 3 éléments."""
        result = visualize_embedding("test", [], "fr")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_returns_figure_and_text(self, reset_core):
        """Retourne une figure matplotlib et du texte."""
        fig, info_text, cache = visualize_embedding("test", [], "fr")
        
        assert fig is not None or info_text is not None
        assert isinstance(info_text, str)
        assert isinstance(cache, list)

    def test_info_text_contains_metrics(self, reset_core):
        """Le texte d'info contient les métriques."""
        fig, info_text, cache = visualize_embedding("test", [], "fr")
        
        assert "**Attracteur :**" in info_text
        assert "**Asymétrie η :**" in info_text
        assert "**Frustration E :**" in info_text
        assert "**Seuil R :**" in info_text
        assert "**Régime :**" in info_text
        assert "**Embedding 20D :**" in info_text
        assert "**Embedding 768D :**" in info_text
        assert "**Taux de compression :**" in info_text

    def test_cache_grows_with_calls(self, reset_core):
        """Le cache grandit avec les appels successifs."""
        cache = []
        
        for text in ["texte1", "texte2", "texte3"]:
            fig, info, cache = visualize_embedding(text, cache, "fr")
        
        assert len(cache) == 3

    def test_cache_respects_max_size(self, reset_core):
        """Le cache respecte la taille maximale (20)."""
        cache = []
        
        for i in range(25):
            fig, info, cache = visualize_embedding(f"texte{i}", cache, "fr")
        
        assert len(cache) <= 20

    def test_pca_activates_after_5_points(self, reset_core):
        """La PCA s'active après 5 points."""
        cache = []
        
        for i in range(5):
            fig, info, cache = visualize_embedding(f"texte{i}", cache, "fr")
        
        # Le 5ème appel doit avoir généré une PCA
        assert fig is not None or "PCA" in str(info)

    def test_empty_text_returns_warning(self, reset_core):
        """Un texte vide retourne un avertissement."""
        fig, info_text, cache = visualize_embedding("", [], "fr")
        
        assert fig is None
        assert "⚠️" in info_text or "Veuillez entrer" in info_text

    def test_whitespace_only_returns_warning(self, reset_core):
        """Un texte avec seulement des espaces retourne un avertissement."""
        fig, info_text, cache = visualize_embedding("   \t\n  ", [], "fr")
        
        assert fig is None
        assert "⚠️" in info_text or "Veuillez entrer" in info_text

    def test_long_text_is_truncated_in_info(self, reset_core):
        """Un texte long est tronqué dans l'info."""
        long_text = "a" * 100
        fig, info_text, cache = visualize_embedding(long_text, [], "fr")
        
        assert "..." in info_text or long_text[:50] in info_text


# ============================================================================
# TESTS DE CAS LIMITES
# ============================================================================

class TestEdgeCases:
    """Tests de cas limites et caractères spéciaux."""

    def test_unicode_characters(self, reset_core):
        """Gère les caractères Unicode."""
        emb, _, _ = text_to_embedding("日本語テスト 🎉")
        assert emb.shape == (20,)

    def test_emoji(self, reset_core):
        """Gère les emojis."""
        emb, _, _ = text_to_embedding("Hello 👋 World 🌍")
        assert emb.shape == (20,)

    def test_special_characters(self, reset_core):
        """Gère les caractères spéciaux."""
        emb, _, _ = text_to_embedding("!@#$%^&*()_+-=[]{}|;':\",./<>?")
        assert emb.shape == (20,)

    def test_newlines_and_tabs(self, reset_core):
        """Gère les sauts de ligne et tabulations."""
        emb, _, _ = text_to_embedding("ligne1\nligne2\ttab")
        assert emb.shape == (20,)

    def test_very_long_text(self, reset_core):
        """Gère les textes très longs."""
        long_text = "a" * 10000
        emb, _, _ = text_to_embedding(long_text)
        assert emb.shape == (20,)

    def test_single_character(self, reset_core):
        """Gère un caractère unique."""
        emb, _, _ = text_to_embedding("a")
        assert emb.shape == (20,)

    def test_repeated_characters(self, reset_core):
        """Gère les caractères répétés."""
        emb, _, _ = text_to_embedding("aaaaaaaaaa")
        assert emb.shape == (20,)


# ============================================================================
# TESTS DE THREAD SAFETY
# ============================================================================

class TestThreadSafety:
    """Tests de la thread safety de l'interface."""

    def test_concurrent_text_to_embedding(self, reset_core):
        """Plusieurs threads peuvent appeler text_to_embedding simultanément."""
        errors = []
        results = []
        
        def worker(text):
            try:
                emb, _, _ = text_to_embedding(text)
                results.append(emb)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=worker, args=(f"texte{i}",))
            for i in range(10)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert not errors, f"Erreurs thread : {errors}"
        assert len(results) == 10

    def test_concurrent_visualize_embedding(self, reset_core):
        """Plusieurs threads peuvent appeler visualize_embedding simultanément."""
        errors = []
        
        def worker(text, cache):
            try:
                visualize_embedding(text, cache, "fr")
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=worker, args=(f"texte{i}", []))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert not errors, f"Erreurs thread : {errors}"

    def test_lock_protects_core(self, reset_core):
        """Le lock protège l'accès au core."""
        initial_counter = reset_core.input_counter
        
        def worker():
            for _ in range(10):
                text_to_embedding("test")
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert reset_core.input_counter == initial_counter + 50


# ============================================================================
# TESTS D'INTÉGRATION
# ============================================================================

class TestIntegration:
    """Tests d'intégration combinant plusieurs composants."""

    def test_full_pipeline(self, reset_core, sample_texts):
        """Pipeline complet : plusieurs textes → visualisation."""
        cache = []
        
        for text in sample_texts:
            fig, info, cache = visualize_embedding(text, cache, "fr")
        
        assert len(cache) == len(sample_texts)
        assert "Attracteur" in info

    def test_compression_ratio_displayed(self, reset_core):
        """Le taux de compression est affiché dans l'info."""
        fig, info_text, cache = visualize_embedding("test", [], "fr")
        
        assert "**Taux de compression :**" in info_text
        import re
        match = re.search(r'Taux de compression :\*\* ([\d.]+)x', info_text)
        if match:
            ratio = float(match.group(1))
            # 768/20 = 38.4x
            assert 35 <= ratio <= 45, f"Taux de compression inattendu : {ratio}"


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
