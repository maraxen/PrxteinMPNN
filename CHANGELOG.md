# Changelog

All notable changes to PrxteinMPNN will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive validation test suite (`tests/validation/`) to monitor key performance metrics
  - Sequence recovery validation across multiple structures
  - Conditional scoring accuracy validation
  - Sampling diversity validation at different temperatures
- Tied position functionality for grouped residue decoding
- Enhanced protein concatenation with globally unique chain IDs and structure mapping
- Extensive test coverage for STE optimization (`tests/sampling/test_ste_optimize.py`)

### Fixed
- Improved autoregressive mask generation for tied position groups
- Enhanced sequence embedding handling in autoregressive sampling
- Fixed edge feature concatenation to properly gather neighbor features

### Changed
- Refactored model architecture to use Equinox modules throughout
- Improved type annotations and JAX typing
- Enhanced test organization and structure

### Performance
- Model architecture validated against ProteinMPNN reference implementation
- Core functionality tested and verified:
  - Encoder properly uses neighbor features (h_j)
  - Decoder contexts correctly structured for unconditional and conditional modes
  - Autoregressive sampling properly implements masked decoding
  - Scale factor of 30.0 correctly applied throughout

### Testing
- Added metric-based validation tests ready for execution
- 20+ unit tests passing for core sampling and scoring functionality
- Validated tied position functionality
- Comprehensive test coverage for key model components

## [0.1.0] - Initial Release

### Added
- Initial JAX-based implementation of ProteinMPNN
- Functional interface for protein sequence design
- Support for unconditional, conditional, and autoregressive sampling
- Integration with Hugging Face Hub for model weights
- Modular architecture with clean separation of concerns
- Comprehensive documentation and examples

---

**Note**: This changelog documents changes made during validation and cleanup of the PrxteinMPNN implementation. Earlier development history is available in git commit logs.
