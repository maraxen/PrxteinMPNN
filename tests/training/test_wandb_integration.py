import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from prxteinmpnn.training.specs import TrainingSpecification
from prxteinmpnn.training.trainer import train
from prxteinmpnn.training.metrics import TrainingMetrics, EvaluationMetrics

class TestWandbIntegration(unittest.TestCase):
    @patch("prxteinmpnn.training.trainer.wandb")
    @patch("prxteinmpnn.training.trainer.create_protein_dataset")
    @patch("prxteinmpnn.training.trainer._init_checkpoint_and_model")
    @patch("prxteinmpnn.training.trainer.create_optimizer")
    @patch("prxteinmpnn.training.trainer.train_step")
    @patch("prxteinmpnn.training.trainer.eval_step")
    @patch("prxteinmpnn.training.trainer.save_checkpoint")
    @patch("prxteinmpnn.training.trainer.eqx.filter_jit")
    def test_wandb_training_flow(
        self,
        mock_jit,
        mock_save_checkpoint,
        mock_eval_step,
        mock_train_step,
        mock_create_optimizer,
        mock_init_ckpt,
        mock_create_dataset,
        mock_wandb
    ):
        # Setup Mocks

        # mock_jit should return the function itself (identity)
        mock_jit.side_effect = lambda f: f

        # Mock Model and State
        mock_model = MagicMock()
        mock_opt_state = MagicMock()
        mock_ckpt_manager = MagicMock()
        mock_perm_manager = MagicMock()

        mock_init_ckpt.return_value = (
            mock_model,
            mock_opt_state,
            0, # start_step
            mock_ckpt_manager,
            mock_perm_manager
        )

        mock_create_optimizer.return_value = (MagicMock(), MagicMock())

        # Mock Data Loader
        # Create a dummy batch
        batch = MagicMock()
        batch.coordinates = jnp.zeros((1, 10, 4, 3))
        batch.mask = jnp.ones((1, 10))
        batch.residue_index = jnp.arange(10)[None, :]
        batch.chain_index = jnp.zeros((1, 10))
        batch.aatype = jnp.zeros((1, 10))
        batch.physics_features = None

        mock_loader = [batch] # One batch
        mock_create_dataset.return_value = mock_loader # create_protein_dataset returns a single loader

        # Mock Step Returns
        train_metrics = TrainingMetrics(
            loss=jnp.array(0.5),
            accuracy=jnp.array(0.9),
            perplexity=jnp.array(1.2),
            learning_rate=jnp.array(0.001),
            grad_norm=jnp.array(0.1)
        )
        mock_train_step.return_value = (mock_model, mock_opt_state, train_metrics)

        eval_metrics = EvaluationMetrics(
            val_loss=jnp.array(0.4),
            val_accuracy=jnp.array(0.95),
            val_perplexity=jnp.array(1.1)
        )
        mock_eval_step.return_value = eval_metrics

        # Create Spec
        spec = TrainingSpecification(
            inputs="dummy/path",
            validation_data="dummy/val",
            num_epochs=1,
            batch_size=1,
            use_wandb=True,
            wandb_project="test_project",
            log_every=1,
            eval_every=1,
            checkpoint_dir="dummy_ckpt",
            use_sharding=False,
            shard_batch=False
        )

        # Run Train
        result = train(spec)

        # Verify WandB Calls
        mock_wandb.init.assert_called_once()
        args, kwargs = mock_wandb.init.call_args
        self.assertEqual(kwargs['project'], "test_project")
        self.assertTrue(kwargs['config']['use_wandb'])

        # Verify Logging
        # Should log training metrics
        self.assertTrue(mock_wandb.log.called)

        # Check that we logged something with "train/loss"
        log_calls = mock_wandb.log.call_args_list

        # We expect at least one log call for training and one for validation
        train_log_found = False
        val_log_found = False

        for call in log_calls:
            args, kwargs = call
            data = args[0]
            if "train/loss" in data:
                train_log_found = True
                self.assertEqual(data["train/loss"], 0.5)
            if "val/loss" in data:
                val_log_found = True
                self.assertAlmostEqual(data["val/loss"], 0.4, places=4)

        self.assertTrue(train_log_found, "Training metrics not logged")
        self.assertTrue(val_log_found, "Validation metrics not logged")

        # Verify Finish
        mock_wandb.finish.assert_called_once()
