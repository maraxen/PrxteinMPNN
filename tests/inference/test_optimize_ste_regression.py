"""Regression tests for optimize_ste module.

This test focuses on the stage_set=None fallback path which uses LOGIT_STRATEGIES
(which was missing an import, causing NameError at runtime).

Regression: Bug #3183 - LOGIT_STRATEGIES not imported in optimize_ste.py
"""

from __future__ import annotations

import jax.numpy as jnp

from aminx.inference.optimize_ste import make_optimize_sequence_fn


class TestOptimizeSteLOGITStrategiesImport:
    """Test that LOGIT_STRATEGIES is properly imported and accessible.

    Regression: Bug #3183 - LOGIT_STRATEGIES missing import would cause NameError
    at line 302 of optimize_ste.py when the stage_set=None fallback path is executed.
    """

    def test_logit_strategies_directly_accessible(self):
        """LOGIT_STRATEGIES should be accessible from optimize_ste module.

        Regression: Bug #3183 - missing import caused NameError at runtime.
        """
        from aminx.inference import optimize_ste

        # Verify LOGIT_STRATEGIES can be imported/accessed
        assert hasattr(optimize_ste, "LOGIT_STRATEGIES")
        logit_strategies = optimize_ste.LOGIT_STRATEGIES
        assert logit_strategies is not None

    def test_logit_strategies_can_get_arithmetic_mean(self):
        """LOGIT_STRATEGIES.get("arithmetic_mean") should work.

        This is what the code at line 302-308 of optimize_ste.py calls.
        """
        from aminx.inference import optimize_ste

        strategies = optimize_ste.LOGIT_STRATEGIES
        strategy_cls = strategies.get("arithmetic_mean")
        assert strategy_cls is not None

    def test_logit_strategies_can_get_geometric_mean(self):
        """LOGIT_STRATEGIES.get("geometric_mean") should work.

        This is one of the fallback paths in the stage_set=None branch.
        """
        from aminx.inference import optimize_ste

        strategies = optimize_ste.LOGIT_STRATEGIES
        strategy_cls = strategies.get("geometric_mean")
        assert strategy_cls is not None

    def test_logit_strategies_can_get_product(self):
        """LOGIT_STRATEGIES.get("product") should work.

        This is another fallback path in the stage_set=None branch.
        """
        from aminx.inference import optimize_ste

        strategies = optimize_ste.LOGIT_STRATEGIES
        strategy_cls = strategies.get("product")
        assert strategy_cls is not None

    def test_logit_strategies_instantiation(self):
        """LOGIT_STRATEGIES strategy classes should be instantiable.

        Verifies that the retrieved strategy classes can be used to create
        instances (which is what line 309 of optimize_ste.py does).
        """
        from aminx.inference import optimize_ste

        strategies = optimize_ste.LOGIT_STRATEGIES
        state_weights = jnp.ones(2) / 2

        # Test arithmetic_mean instantiation
        strategy_cls = strategies.get("arithmetic_mean")
        strategy_instance = strategy_cls(state_weights)
        assert strategy_instance is not None

        # Test geometric_mean instantiation
        strategy_cls = strategies.get("geometric_mean")
        strategy_instance = strategy_cls(state_weights)
        assert strategy_instance is not None

        # Test product instantiation
        strategy_cls = strategies.get("product")
        strategy_instance = strategy_cls(state_weights)
        assert strategy_instance is not None
