# Contributing to PrxteinMPNN

We welcome contributions from the community! To ensure a smooth development process, please follow these guidelines.

## Setting Up Your Development Environment

To get started, you'll need `uv` and `pre-commit` installed.

1.  **Install `uv`**:
    Follow the official installation instructions for `uv` from [astral.sh](https://astral.sh/uv).

2.  **Clone the repository**:
    ```bash
    git clone https://github.com/maraxen/PrxteinMPNN.git
    cd PrxteinMPNN
    ```

3.  **Install dependencies**:
    Create a virtual environment and install the necessary dependencies using `uv`.
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -e ".[dev]"
    ```

4.  **Set up pre-commit hooks**:
    Install the pre-commit hooks to ensure your contributions adhere to our code quality standards.
    ```bash
    pre-commit install
    ```

## Running Pre-commit Hooks

The pre-commit hooks will run automatically when you commit your changes. To run them manually on all files, use the following command:

```bash
pre-commit run --all-files
```

This will run `ruff` for linting and formatting, and `ty` for type checking. Please ensure that all hooks pass before submitting a pull request.

## A Note on Type Checking

We use `astral ty` for static type checking. Currently, the `ty` pre-commit hook is configured to skip the `tests/` directory due to a number of existing type errors.

We welcome contributions to help make our test suite fully type-compliant! If you'd like to help, you can run `ty` on the `tests/` directory and submit a pull request with your fixes.

## Submitting a Pull Request

1.  Create a new branch for your feature or bug fix.
2.  Make your changes and commit them.
3.  Ensure all pre-commit hooks pass.
4.  Push your branch to your fork.
5.  Open a pull request with a clear description of your changes.

Thank you for contributing to PrxteinMPNN!