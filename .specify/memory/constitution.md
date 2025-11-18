<!--
Sync Impact Report:
- Version change: 1.0.0 → 1.0.1 (PATCH: clarification of Principle II)
- Modified principles: II. Type Safety & Static Analysis (updated to reference make format-all)
- Added sections: N/A
- Removed sections: N/A
- Templates requiring updates:
  ✅ plan-template.md - Constitution Check section exists and aligns
  ✅ spec-template.md - No constitution-specific references to update
  ✅ tasks-template.md - No constitution-specific references to update
  ✅ command files - All references verified (speckit.plan.md, speckit.analyze.md, speckit.constitution.md)
- Follow-up TODOs: None
-->

# ctrlf Constitution

## Core Principles

### I. Test-First Development (NON-NEGOTIABLE)

All features MUST follow Test-Driven Development (TDD) methodology. Tests MUST be written before implementation, approved by stakeholders, and verified to fail before code is written. The Red-Green-Refactor cycle is strictly enforced. All code MUST have corresponding tests with coverage requirements enforced by CI/CD.

**Rationale**: Ensures code correctness, prevents regressions, and maintains high code quality standards throughout the project lifecycle.

### II. Type Safety & Static Analysis

All code MUST pass `make format-all` which enforces type checking (mypy with strict mode enabled) and static analysis (ruff, pylint via pre-commit/prek). Type hints are mandatory for all function signatures, class attributes, and module-level variables. All linting and type checking MUST pass without errors before code can be merged.

**Rationale**: Type safety catches errors at development time, improves code maintainability, and enhances IDE support for developers. Using `make format-all` ensures consistent enforcement across all code quality checks.

### III. CLI Interface Standard

All functionality MUST be accessible via command-line interface using Typer. Commands MUST follow consistent patterns: structured output (JSON when appropriate), clear error messages to stderr, and human-readable formats. All scripts MUST be executable and properly documented.

**Rationale**: CLI interfaces ensure the library is usable in automation, scripts, and various deployment scenarios without requiring Python imports.

### IV. Data Integrity & Validation

All data processing operations MUST use Pandera for schema validation when working with Polars DataFrames. Data transformations MUST validate input schemas and output schemas. Invalid data MUST be rejected with clear error messages.

**Rationale**: Data validation prevents silent failures, ensures data quality, and provides clear feedback when data doesn't meet expected schemas.

### V. Observability & Logging

All operations MUST use structured logging via structlog. Logs MUST include contextual information (request IDs, operation names, timing). Log levels MUST be appropriate (DEBUG for development, INFO for operations, ERROR for failures). Critical operations MUST be logged with sufficient detail for debugging.

**Rationale**: Structured logging enables effective debugging, monitoring, and operational visibility in production environments.

## Development Standards

### Code Quality

- All code MUST pass `make format-all` before commit (runs ruff, pylint, mypy via pre-commit/prek)
- Code MUST follow PEP 8 style guidelines (enforced by ruff)
- Maximum line length: 100 characters (configurable exceptions for URLs, long strings)
- All public APIs MUST have docstrings following Google or NumPy style
- Private functions and classes MUST have inline comments explaining non-obvious logic

### Testing Requirements

- Unit tests MUST cover all business logic with >80% coverage
- Integration tests MUST cover all CLI commands and data processing workflows
- Contract tests MUST validate data schemas and API contracts
- Tests MUST be deterministic and not depend on external services or timing
- All tests MUST pass in CI/CD before merge

### Documentation

- README.md MUST include installation, basic usage, and examples
- All public functions, classes, and modules MUST have docstrings
- Complex algorithms or business logic MUST have inline comments
- Breaking changes MUST be documented in CHANGELOG.md

## Governance

This constitution supersedes all other development practices and guidelines. All pull requests and code reviews MUST verify compliance with these principles. Amendments to this constitution require:

1. Documentation of the proposed change and rationale
2. Review and approval by project maintainers
3. Update to version number following semantic versioning:
   - **MAJOR**: Backward incompatible governance changes or principle removals
   - **MINOR**: New principles added or existing principles materially expanded
   - **PATCH**: Clarifications, wording improvements, typo fixes
4. Propagation of changes to all dependent templates and documentation
5. Migration plan if the change affects existing code or workflows

Complexity additions MUST be justified. When a principle violation is necessary, it MUST be documented in the implementation plan with rationale and simpler alternatives considered.

**Version**: 1.0.1 | **Ratified**: 2025-11-12 | **Last Amended**: 2025-11-18
