import pytest
from unittest.mock import Mock, patch
from io import StringIO

from prxteinmpnn.run.prep import prep_protein_stream_and_model


@pytest.fixture
def mock_run_spec():
  """Create a mock RunSpecification for testing.
  
  Returns:
    Mock: A mock RunSpecification object with all required attributes.
  """
  spec = Mock()
  spec.chain_id = "A"
  spec.model = "ca_only"
  spec.altloc = "A"
  spec.inputs = ["test_input.pdb"]
  spec.batch_size = 1
  spec.foldcomp_database = None
  spec.num_workers = 1
  spec.model_version = "v_48_020"
  spec.model_weights = None
  return spec


@pytest.fixture
def mock_protein_iterator():
  """Create a mock protein iterator for testing.
  
  Returns:
    Mock: A mock IterDataset object.
  """
  return Mock()


@pytest.fixture
def mock_model_parameters():
  """Create a mock model parameters object for testing.
  
  Returns:
    Mock: A mock ModelParameters object.
  """
  return Mock()


@patch("prxteinmpnn.run.prep.get_mpnn_model")
@patch("prxteinmpnn.run.prep.loaders.create_protein_dataset")
def testprep_protein_stream_and_model_basic(
  mock_create_dataset, mock_get_model, mock_run_spec, mock_protein_iterator, mock_model_parameters
):
  """Test basic functionality of prep_protein_stream_and_model.
  
  Args:
    mock_create_dataset: Mock for loaders.create_protein_dataset.
    mock_get_model: Mock for get_mpnn_model.
    mock_run_spec: Mock RunSpecification fixture.
    mock_protein_iterator: Mock protein iterator fixture.
    mock_model_parameters: Mock model parameters fixture.
    
  Returns:
    None
    
  Raises:
    AssertionError: If the function doesn't behave as expected.
  """
  # Setup mocks
  mock_create_dataset.return_value = mock_protein_iterator
  mock_get_model.return_value = mock_model_parameters
  
  # Call function
  result_iterator, result_params = prep_protein_stream_and_model(mock_run_spec)
  
  # Verify loaders.create_protein_dataset was called correctly
  mock_create_dataset.assert_called_once_with(
    ["test_input.pdb",],
    batch_size=1,
    foldcomp_database=None,
    parse_kwargs={
      "chain_id": "A",
      "model": "ca_only",
      "altloc": "A",
    },
    num_workers=1,
  )
  
  # Verify get_mpnn_model was called correctly
  mock_get_model.assert_called_once_with(
    model_version="v_48_020",
    model_weights=None,
  )
  
  # Verify return values
  assert result_iterator is mock_protein_iterator
  assert result_params is mock_model_parameters


@patch("prxteinmpnn.run.prep.get_mpnn_model")
@patch("prxteinmpnn.run.prep.loaders.create_protein_dataset")
def testprep_protein_stream_and_model_multiple_inputs(
  mock_create_dataset, mock_get_model, mock_protein_iterator, mock_model_parameters
):
  """Test prep_protein_stream_and_model with multiple input files.
  
  Args:
    mock_create_dataset: Mock for loaders.create_protein_dataset.
    mock_get_model: Mock for get_mpnn_model.
    mock_protein_iterator: Mock protein iterator fixture.
    mock_model_parameters: Mock model parameters fixture.
    
  Returns:
    None
    
  Raises:
    AssertionError: If the function doesn't handle multiple inputs correctly.
  """
  # Setup spec with multiple inputs
  spec = Mock()
  spec.chain_id = "B"
  spec.model = "full_atom"
  spec.altloc = "B"
  spec.inputs = ["file1.pdb", "file2.pdb", "file3.pdb"]
  spec.batch_size = 4
  spec.foldcomp_database = "/path/to/db"
  spec.num_workers = 2
  spec.model_version = "v_48_030"
  spec.model_weights = "/path/to/weights.pt"
  
  # Setup mocks
  mock_create_dataset.return_value = mock_protein_iterator
  mock_get_model.return_value = mock_model_parameters
  
  # Call function
  result_iterator, result_params = prep_protein_stream_and_model(spec)
  
  # Verify loaders.create_protein_dataset was called with multiple inputs
  mock_create_dataset.assert_called_once_with(
    ["file1.pdb", "file2.pdb", "file3.pdb"],
    batch_size=4,
    foldcomp_database="/path/to/db",
    parse_kwargs={
      "chain_id": "B",
      "model": "full_atom",
      "altloc": "B",
    },
    num_workers=2,
  )
  
  # Verify get_mpnn_model was called correctly
  mock_get_model.assert_called_once_with(
    model_version="v_48_030",
    model_weights="/path/to/weights.pt",
  )
  
  # Verify return values
  assert result_iterator is mock_protein_iterator
  assert result_params is mock_model_parameters


@patch("prxteinmpnn.run.prep.get_mpnn_model")
@patch("prxteinmpnn.run.prep.loaders.create_protein_dataset")
def testprep_protein_stream_and_model_stringio_input(
  mock_create_dataset, mock_get_model, mock_protein_iterator, mock_model_parameters
):
  """Test prep_protein_stream_and_model with StringIO input.
  
  Args:
    mock_create_dataset: Mock for loaders.create_protein_dataset.
    mock_get_model: Mock for get_mpnn_model.
    mock_protein_iterator: Mock protein iterator fixture.
    mock_model_parameters: Mock model parameters fixture.
    
  Returns:
    None
    
  Raises:
    AssertionError: If the function doesn't handle StringIO input correctly.
  """
  # Setup spec with StringIO input
  stringio_input = StringIO("ATOM      1  N   ALA A   1      20.154  16.967  12.931")
  spec = Mock()
  spec.chain_id = None
  spec.model = "ca_only"
  spec.altloc = None
  spec.inputs = stringio_input
  spec.batch_size = 1
  spec.foldcomp_database = None
  spec.num_workers = 1
  spec.model_version = "v_48_020"
  spec.model_weights = None
  
  # Setup mocks
  mock_create_dataset.return_value = mock_protein_iterator
  mock_get_model.return_value = mock_model_parameters
  
  # Call function
  result_iterator, result_params = prep_protein_stream_and_model(spec)
  
  # Verify loaders.create_protein_dataset was called with tuple containing StringIO
  mock_create_dataset.assert_called_once_with(
    (stringio_input,),
    batch_size=1,
    foldcomp_database=None,
    parse_kwargs={
      "chain_id": None,
      "model": "ca_only",
      "altloc": None,
    },
    num_workers=1,
  )
  
  # Verify get_mpnn_model was called correctly
  mock_get_model.assert_called_once_with(
    model_version="v_48_020",
    model_weights=None,
  )
  
  # Verify return values
  assert result_iterator is mock_protein_iterator
  assert result_params is mock_model_parameters


@patch("prxteinmpnn.run.prep.get_mpnn_model")
@patch("prxteinmpnn.run.prep.loaders.create_protein_dataset")
def testprep_protein_stream_and_model_empty_inputs(
  mock_create_dataset, mock_get_model, mock_protein_iterator, mock_model_parameters
):
  """Test prep_protein_stream_and_model with empty inputs list.
  
  Args:
    mock_create_dataset: Mock for loaders.create_protein_dataset.
    mock_get_model: Mock for get_mpnn_model.
    mock_protein_iterator: Mock protein iterator fixture.
    mock_model_parameters: Mock model parameters fixture.
    
  Returns:
    None
    
  Raises:
    AssertionError: If the function doesn't handle empty inputs correctly.
  """
  # Setup spec with empty inputs
  spec = Mock()
  spec.chain_id = "A"
  spec.model = "ca_only"
  spec.altloc = "A"
  spec.inputs = []
  spec.batch_size = 1
  spec.foldcomp_database = None
  spec.num_workers = 1
  spec.model_version = "v_48_020"
  spec.model_weights = None
  
  # Setup mocks
  mock_create_dataset.return_value = mock_protein_iterator
  mock_get_model.return_value = mock_model_parameters
  
  # Call function
  result_iterator, result_params = prep_protein_stream_and_model(spec)
  
  # Verify loaders.create_protein_dataset was called with empty list
  mock_create_dataset.assert_called_once_with(
    [],
    batch_size=1,
    foldcomp_database=None,
    parse_kwargs={
      "chain_id": "A",
      "model": "ca_only",
      "altloc": "A",
    },
    num_workers=1,
  )
  
  # Verify get_mpnn_model was called correctly
  mock_get_model.assert_called_once_with(
    model_version="v_48_020",
    model_weights=None,
  )
  
  # Verify return values
  assert result_iterator is mock_protein_iterator
  assert result_params is mock_model_parameters