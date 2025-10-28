"""Unit tests for K-Means clustering functionality."""

import chex
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.ensemble.kmeans import kmeans, _kmeans_plusplus_init as kmeans_plusplus_initialization


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    key = jax.random.PRNGKey(42)
    return jax.random.normal(key, (100, 5))


@pytest.fixture
def small_data():
    """Create small sample data for detailed testing."""
    return jnp.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [0.1, 0.1],
            [1.1, 1.1],
            [2.1, 2.1],
        ]
    )


class TestKMeansPlusPlusInit:
    """Test the K-Means++ initialization function."""

    def test_correct_output_shape(self, sample_data):
        """Test that K-Means++ returns correct centroid shape."""
        key = jax.random.PRNGKey(42)
        n_clusters = 3

        centroids = kmeans_plusplus_initialization(key, sample_data, n_clusters)

        chex.assert_shape(centroids, (n_clusters, sample_data.shape[1]))

    def test_centroids_are_data_points(self, small_data):
        """Test that initial centroids are actual data points."""
        key = jax.random.PRNGKey(42)
        n_clusters = 3

        centroids = kmeans_plusplus_initialization(key, small_data, n_clusters)

        # Each centroid should be one of the original data points
        for centroid in centroids:
            distances = jnp.linalg.norm(small_data - centroid, axis=1)
            assert jnp.min(distances) < 1e-10  # Should match exactly

    def test_single_cluster(self, sample_data):
        """Test K-Means++ with single cluster."""
        key = jax.random.PRNGKey(42)

        centroids = kmeans_plusplus_initialization(key, sample_data, 1)

        chex.assert_shape(centroids, (1, sample_data.shape[1]))

    def test_deterministic_with_same_key(self, sample_data):
        """Test that same key produces same initialization."""
        key = jax.random.PRNGKey(42)
        n_clusters = 3

        centroids1 = kmeans_plusplus_initialization(key, sample_data, n_clusters)
        centroids2 = kmeans_plusplus_initialization(key, sample_data, n_clusters)

        chex.assert_trees_all_close(centroids1, centroids2)

    def test_different_keys_different_results(self, sample_data):
        """Test that different keys produce different initializations."""
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(24)
        n_clusters = 3

        centroids1 = kmeans_plusplus_initialization(key1, sample_data, n_clusters)
        centroids2 = kmeans_plusplus_initialization(key2, sample_data, n_clusters)

        # Results should be different (with high probability)
        assert not jnp.allclose(centroids1, centroids2)

    @pytest.mark.parametrize("n_clusters", [1, 2, 3, 5])
    def test_various_cluster_counts(self, sample_data, n_clusters):
        """Test K-Means++ with various cluster counts."""
        key = jax.random.PRNGKey(42)

        centroids = kmeans_plusplus_initialization(key, sample_data, n_clusters)

        chex.assert_shape(centroids, (n_clusters, sample_data.shape[1]))


class TestKMeans:
    """Test the K-Means clustering function."""

    def test_correct_output_shape(self, sample_data):
        """Test that K-Means returns correct label shape."""
        key = jax.random.PRNGKey(42)
        n_clusters = 3

        labels = kmeans(key, sample_data, n_clusters)

        chex.assert_shape(labels, (sample_data.shape[0],))

    def test_labels_in_valid_range(self, sample_data):
        """Test that all labels are in valid range."""
        key = jax.random.PRNGKey(42)
        n_clusters = 3

        labels = kmeans(key, sample_data, n_clusters)

        assert jnp.all(labels >= 0)
        assert jnp.all(labels < n_clusters)

    def test_well_separated_clusters(self):
        """Test K-Means on well-separated clusters."""
        # Create well-separated clusters
        cluster1 = jnp.array([[0.0, 0.0], [0.1, 0.1], [0.0, 0.1]])
        cluster2 = jnp.array([[5.0, 5.0], [5.1, 5.1], [5.0, 5.1]])
        cluster3 = jnp.array([[10.0, 10.0], [10.1, 10.1], [10.0, 10.1]])

        data = jnp.vstack([cluster1, cluster2, cluster3])
        key = jax.random.PRNGKey(42)

        labels = kmeans(key, data, 3)

        # Points in same cluster should have same label
        assert len(jnp.unique(labels[:3])) == 1  # Cluster 1
        assert len(jnp.unique(labels[3:6])) == 1  # Cluster 2
        assert len(jnp.unique(labels[6:9])) == 1  # Cluster 3

    def test_single_cluster(self, sample_data):
        """Test K-Means with single cluster."""
        key = jax.random.PRNGKey(42)

        labels = kmeans(key, sample_data, 1)

        # All points should be in cluster 0
        assert jnp.all(labels == 0)

    @pytest.mark.parametrize("max_iters", [1, 10, 50])
    def test_different_max_iterations(self, sample_data, max_iters):
        """Test K-Means with different maximum iterations."""
        key = jax.random.PRNGKey(42)

        labels = kmeans(key, sample_data, 3, max_iters=max_iters)

        chex.assert_shape(labels, (sample_data.shape[0],))
        assert jnp.all(labels >= 0)
        assert jnp.all(labels < 3)

    def test_deterministic_with_same_key(self, sample_data):
        """Test that same key produces same clustering."""
        key = jax.random.PRNGKey(42)

        labels1 = kmeans(key, sample_data, 3)
        labels2 = kmeans(key, sample_data, 3)

        chex.assert_trees_all_equal(labels1, labels2)