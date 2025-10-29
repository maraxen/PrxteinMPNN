# PrxteinMPNN Agents Instructions

These instructions guide LLMs in understanding the codebase, adhering to best practices, and contributing effectively to the "PrxteinMPNN" project.

## 1. Project Overview and Goals

This project focuses on **implementing a functional interface for ProteinMPNN**, leveraging the **JAX** ecosystem for accelerated computation.

**Primary Goals:**

* **Increased Transparency:** Provide a clear and functional interface for ProteinMPNN, enabling users to easily understand and utilize the model.
* **JAX Compatibility:** Ensure all components are JAX-compatible, allowing for efficient computation and integration with JAX's functional programming paradigm.
* **Modular Design:** Maintain a modular structure to facilitate easy updates and extensions to the model.
* **Performance Optimization:** Utilize JAX's capabilities (e.g., JIT compilation, vectorization) to optimize performance for large-scale protein design tasks.

**Key Features:**

## 2. Core Principles & Development Practices

Adherence to these principles is paramount for maintaining code quality and consistency:

### A. JAX Idioms and Functional Programming

* **Prioritize JAX-compatible code:** All new code, especially numerical operations, should be written with JAX's functional programming paradigm in mind.
* **Immutability:** Favor immutable data structures (e.g., JAX PyTrees, `equinox.Module`, `flax.linen.Module`) where applicable.
* **JIT/Vmap/Scan Compatibility:** Ensure functions are compatible with JAX's `jit`, `vmap`, and `scan` transformations for performance.
* **Static Arguments:** Utilize `static_argnums` for function arguments that are not JAX types and do not change across JAX transformations (e.g., Python built-in types, strings, tuples of static values, or dataclasses acting as auxiliary data).
* **Documentation:** Use Google-style docstrings for all functions, including type hints and examples. Ensure that JAX transformations are clearly documented, especially when using `jit`, `vmap`, or `scan`.
* **Example  Docstring:**

```python
def example_function():
    """Test the example function with a specific input.
    Args:
        None
    Returns:
        None
    Raises:
        TypeError: If the output does not match the expected value.
    Example:
        >>> example_function()
    """
    result = example_function(input_data)
    expected = expected_output
    if not isinstance(result, expected_type):
        raise TypeError(f"Expected {expected_type}, but got {type(result)}")
```

### B. Code Quality & Linting (Ruff)

* **Linter:** Use **Ruff** for linting.
* **Configuration:** Adhere to the `ruff.toml` settings.
  * `select = ["ALL"]` (all rules enabled by default)
  * `ignore = ["PD"]` (pandas-specific rules are ignored)
  * **`line-length = 100`**: This is the primary target for line length. Ensure generated code adheres to this limit.
  * `indent-width = 2`
  * `fix = true` (Ruff's autofix capabilities should be utilized).
* **Execution:** Run `ruff check src/ --fix` regularly to apply automatic fixes.
* **Fix Failure Threshold:** **If automated `ruff --fix` attempts fail more than 5 times consecutively on the same set of issues, cease further attempts and flag the code for manual review by the user.**

### C. Type Checking (Pyright)

* **Evaluator:** Use **Pyright** for static type checking.
* **Strict Mode:** All code should pass Pyright in **strict mode**. Ensure type hints are accurate and comprehensive to satisfy strict type checking.

### D. Testing

* **Availability:** All new components and features *must* be accompanied by comprehensive **unit and/or integration tests**.
* **Framework:** Use `pytest` and where relevant`chex`
* **Test Location:** Place tests in the `tests/` directory, mirroring the structure of the source code (e.g., `tests/models/`, `tests/sampling/`, etc.).
* **Execution:** Tests are located in the `tests/` directory. Run tests with `uv run pytest tests/`.
* **Python Commands:** Always use `uv run python` instead of `python` or `python3` to ensure the correct environment is used.
* **Test Philosophy:** Tests should cover typical use cases, edge cases, and ensure correctness of JAX transformations where applicable.
* **Test Coverage:** Aim for high test coverage, especially for critical components like SMC steps, resampling methods, and scoring functions.
* **Test Structure:** Organize tests by functionality (e.g., `sampling/`, `scoring/`, `utils/`) and ensure they mirror the structure of the source code for clarity.
* **Test Documentation:** Use Google-style docstrings for tests, clearly describing the purpose, inputs, and expected outputs.
* **Example Test Docstring:**

```python
def test_example_function():
    """Test the example function with a specific input.
    Args:
        None
    Returns:
        None
    Raises:
        AssertionError: If the output does not match the expected value. 
    Example:
        >>> test_example_function()
    """
    result = example_function(input_data)
    expected = expected_output
    assert result == expected, f"Expected {expected}, but got {result}"
```

* **Test Coverage Reporting:** Use `pytest-cov` to generate coverage reports. Aim for at least 90% coverage across the codebase.
* **Test Failures:** If a test fails, provide a clear error message indicating the expected vs. actual output, and ensure the test is reproducible.
* **Test Data:** Use fixtures or mock data where necessary to ensure tests are deterministic and do not rely on external state.
* **Test Dependencies:** Ensure that tests do not introduce unnecessary dependencies. Use `pytest` fixtures to manage setup and teardown of test environments.
* **Test Isolation:** Each test should be independent and not rely on the state left by previous tests. Use fixtures to set up and tear down any necessary state.
* **Test Performance:** Ensure tests run efficiently, especially for performance-critical components. Use `pytest-benchmark` or the JAX equivalent `jax.benchmark` for performance testing where applicable.

### E. Code Structure & Modularity

* **Modularity:** Maintain the modular structure within the `src/prxteinmpnn/` directory (e.g., `sampling/`, `scoring/`, `utils/`).
* **DRY Principle:** Adhere to the "Don't Repeat Yourself" (DRY) principle. Refactor common logic into reusable functions or modules.
* **Dependencies:** Avoid adding unnecessary external dependencies. Keep `requirements.txt` minimal and up-to-date.
* **Documentation:** Ensure all functions, classes, and complex logic are well-documented with clear docstrings and comments.

## 3. Current Focus & Known Issues

* **Current Focus:** Finalizing the functional interface for ProteinMPNN, ensuring all components are JAX-compatible and adhere to the project's core principles.
* **Known Issues:**
* Lack of comprehensive tests
* The tests in `tests/ensemble/test_em_fit.py` and `tests/ensemble/test_gmm.py` are currently backlogged due to issues with mock objects and JAX's JIT compilation.

## 4. Interaction Guidelines

* **Prioritize JAX-idiomatic solutions, including careful consideration of `static_argnums` and dataclass PyTree registration.**
* **Strictly adhere to Ruff linting rules and Pyright strict mode for type checking.**
* **Always include appropriate tests for new code.**
* **If automated fixes for linting or type errors fail repeatedly (more than 5 times for the same issue), report to the user for manual review.**
* Refer to `GEMINI.md` for detailed project context and goals.
* Refer to `README.md` for project setup and general overview.

## 5. Example Function Documentation

"""
Instructions:
* Use Google style docstrings for documenting Python functions.
* Google style docstrings include sections such as Args, Returns, Raises, and Examples.
* Each parameter should be documented under the Args section with its type and description.
* The Returns section should describe the return value and its type.
* The Raises section should list possible exceptions raised by the function, if any.
* Optionally, include an Examples section to demonstrate usage.

Args:
  selection (type): Description of the selection parameter.

Returns:
  type: Description of the return value.

Raises:
  ExceptionType: Description of the exception raised (if applicable).

Example:
  >>> result = function_name(selection)
  >>> print(result)
"""
