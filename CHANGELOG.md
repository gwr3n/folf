# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-08-03

### Added

- Initial stable Python library release using `src` layout and `pyproject.toml`.
- Core first-order loss APIs for empirical and complementary first-order loss computation.
- Scalar-product variants including multivariate normal closed-form support.
- Jensen partitioner implementations:
	- Uniform partitioner
	- Minimax partitioner
- Piecewise linearization helpers for standard normal loss approximations.
- Linearization parameter factory for selecting approximation parameters from
	error tolerance and model constants.
- Probability utility modules:
	- Sampling strategies enum (`SRS`, `LHS`)
	- Simple random and Latin hypercube sample generation
	- Multivariate Gaussian CDF approximation utility
- General utilities:
	- SHA-256 helper
	- JSON serialization/deserialization helper
- End-user README with practical usage examples and modeling workflow guidance.
- Development and quality tooling:
	- Ruff lint configuration
	- mypy type-check configuration
	- pytest smoke tests
	- GitHub Actions CI for lint, type-check, and test on Python 3.10-3.13
	- Publish metadata (homepage, repository, issues, changelog links)

### Changed

- README reorganized to prioritize user-focused examples over migration/process
	mapping details.

### Notes

- This release establishes a stable baseline API for the `folf` Python package.
- Internal numeric behavior may still evolve in future releases for performance
	and stability improvements while preserving public API compatibility.
