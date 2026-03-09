# AGENTS.md - Coding Agent Guidelines

This document provides essential information for agentic coding systems operating in this repository. It includes build/test commands, code style guidelines, and operational conventions.

## Project Type Detection

Before making changes, determine the project type by checking for these files:
- **JavaScript/TypeScript**: `package.json`, `tsconfig.json`
- **Python**: `pyproject.toml`, `setup.py`, `requirements.txt`
- **Rust**: `Cargo.toml`
- **Go**: `go.mod`
- **Java**: `pom.xml`, `build.gradle`

## Build Commands

### JavaScript/TypeScript Projects
```bash
# Install dependencies
npm install
# or
yarn install
# or  
pnpm install

# Build project
npm run build
# or
yarn build

# Development server
npm run dev
# or
yarn dev
```

### Python Projects
```bash
# Install dependencies
pip install -e .
# or
poetry install
# or
pip install -r requirements.txt

# Build package
python setup.py build
# or
poetry build

# Run module directly
python -m your_package
```

### Rust Projects
```bash
# Build
cargo build
# Build release
cargo build --release

# Run
cargo run
```

### Go Projects
```bash
# Build
go build ./...
# Install
go install ./...

# Run
go run main.go
```

## Test Commands

### Running All Tests
```bash
# JavaScript/TypeScript
npm test
# or
yarn test

# Python  
pytest
# or
python -m pytest
# or
tox

# Rust
cargo test

# Go
go test ./...
```

### Running Single Tests
**JavaScript/TypeScript (Jest):**
```bash
# Run specific test file
npm test -- path/to/test/file.test.js
# Run tests matching pattern
npm test -- -t "test name pattern"
# Run single test in watch mode
npm test -- --watch path/to/test/file.test.js
```

**JavaScript/TypeScript (Vitest):**
```bash
# Run specific test file
npm test -- path/to/test/file.test.ts
# Run single test
npm test -- -t "test name"
```

**Python (pytest):**
```bash
# Run specific test file
pytest path/to/test_file.py
# Run specific test function
pytest path/to/test_file.py::test_function_name
# Run specific test method in class
pytest path/to/test_file.py::TestClass::test_method
# Run tests matching pattern
pytest -k "pattern"
```

**Rust:**
```bash
# Run specific test
cargo test test_function_name
# Run tests in specific module
cargo test module_name::
```

**Go:**
```bash
# Run specific test file
go test -run TestFunctionName path/to/package
# Run all tests in directory
go test ./path/to/directory
```

## Lint Commands

### JavaScript/TypeScript
```bash
# Check linting
npm run lint
# Fix linting issues
npm run lint -- --fix
# Type check
npm run typecheck
```

### Python
```bash
# Lint with ruff
ruff check .
ruff check --fix .
# Format with black
black .
# Type check with mypy
mypy .
```

### Rust
```bash
# Format code
cargo fmt
# Lint code
cargo clippy
```

### Go
```bash
# Format code
gofmt -w .
# Vet code
go vet ./...
```

## Code Style Guidelines

### General Principles
- **Readability over cleverness**: Code should be immediately understandable
- **Consistency**: Follow existing patterns in the codebase
- **Minimal changes**: Only modify what's necessary for the task
- **No comments unless requested**: Avoid adding explanatory comments
- **Preserve existing formatting**: Don't reformat unrelated code

### Imports and Dependencies
- **JavaScript/TypeScript**: Use ES6 imports, group by type (stdlib, external, internal)
- **Python**: Follow PEP 8 import ordering (stdlib, third-party, local)
- **Rust**: Use explicit imports, avoid glob imports (`use std::collections::*`)
- **Go**: Use standard library imports first, then external, then internal

### Formatting
- **Indentation**: Use 2 spaces for JS/TS/JSON, 4 spaces for Python, tabs for Go/Rust
- **Line length**: Keep lines under 80-100 characters when possible
- **Trailing commas**: Use trailing commas in multiline arrays/objects
- **Semicolons**: Follow language conventions (required in JS/TS/Go, not in Python/Rust)

### Types and Typing
- **TypeScript**: Always use explicit types for function parameters and return values
- **Python**: Use type hints (PEP 484) for all functions and variables
- **Rust**: Leverage the type system fully, avoid excessive type annotations
- **Go**: Use explicit types, leverage interfaces for polymorphism

### Naming Conventions
- **Variables/Functions**: camelCase (JS/TS/Go), snake_case (Python), snake_case (Rust)
- **Constants**: UPPER_SNAKE_CASE (all languages)
- **Classes/Types**: PascalCase (JS/TS/Go), PascalCase (Python), PascalCase (Rust)
- **Private members**: Leading underscore (Python/JS), lowercase (Go/Rust)

### Error Handling
- **JavaScript/TypeScript**: Use try/catch for async operations, throw descriptive errors
- **Python**: Raise specific exception types, include meaningful error messages
- **Rust**: Use Result types properly, provide context with `anyhow` or custom errors
- **Go**: Return error as last parameter, handle errors immediately

### Testing Patterns
- **Unit tests**: Test one thing at a time, use descriptive test names
- **Mocking**: Mock external dependencies, don't mock internal functions
- **Test data**: Use realistic but minimal test data
- **Assertions**: Be specific about expected outcomes

## Security Guidelines

### NEVER Do These
- Log or expose secrets, API keys, or credentials
- Commit sensitive files (.env, credentials.json, etc.)
- Disable security features or validation
- Use eval() or similar dangerous functions
- Implement authentication/authorization logic without proper review

### Always Do These
- Validate and sanitize all inputs
- Use parameterized queries for databases
- Follow principle of least privilege
- Handle errors gracefully without exposing internal details

## File Operations

### Reading Files
- Use appropriate tools (Read, Glob, Grep) instead of bash commands
- Read entire files when possible rather than small chunks
- Preserve exact formatting including whitespace

### Writing/Editing Files
- Always read file before editing
- Preserve existing indentation and formatting exactly
- Use exact string matching for edits
- Never overwrite files without reading first

### Creating Files
- Prefer editing existing files over creating new ones
- Only create files when explicitly required
- Follow existing directory structure patterns

## Git and Version Control

### Commit Guidelines
- Only commit when explicitly requested by user
- Write descriptive commit messages focusing on "why" not "what"
- Never commit files containing secrets
- Verify changes with git diff before committing

### Branch Management
- Don't create branches unless requested
- Work in current branch unless instructed otherwise
- Never force push without explicit user request

## Special Instructions

### Cursor Rules
If `.cursor/rules/` or `.cursorrules` exists, follow those rules precisely. These typically contain:
- Forbidden patterns or anti-patterns
- Required documentation standards  
- Specific architectural constraints
- Language-specific best practices

### GitHub Copilot Instructions
If `.github/copilot-instructions.md` exists, incorporate those guidelines which may include:
- Code generation restrictions
- Documentation requirements
- Testing expectations
- Style preferences

## Language-Specific Notes

### JavaScript/TypeScript
- Prefer const over let/var
- Use async/await over callbacks
- Leverage destructuring for cleaner code
- Avoid any type; use unknown or specific types

### Python
- Follow PEP 8 strictly
- Use f-strings for string formatting (Python 3.6+)
- Leverage context managers for resource handling
- Use pathlib over os.path for file operations

### Rust
- Follow Rust naming conventions strictly
- Use idiomatic error handling with Result/Option
- Leverage ownership and borrowing properly
- Avoid unsafe code unless absolutely necessary

### Go
- Follow Go idioms (errors as values, etc.)
- Use gofmt consistently
- Leverage goroutines and channels appropriately
- Write table-driven tests

## Verification Steps

After making changes, always:
1. Run relevant linters/formatters
2. Run type checker if applicable  
3. Run affected tests
4. Verify no unintended changes were made
5. Ensure build still succeeds

## When in Doubt

- Ask the user for clarification
- Follow existing code patterns in the repository
- Prioritize safety and correctness over completeness
- Err on the side of minimal changes