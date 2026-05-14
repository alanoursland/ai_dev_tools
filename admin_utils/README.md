# Admin Utilities

This directory contains utility scripts for maintaining and documenting the KinoPulse library.

## library_structure.py

Automatically generates API documentation by parsing Python source code using AST.

### Purpose

Creates `library_structure.txt` - a lightweight API reference showing all public classes, methods, and functions across the codebase. This serves as a quick reference for:

- **API verification**: Checking if implementations match design specs
- **Finding methods**: Quickly seeing what's available in each module
- **Integration work**: Understanding available interfaces between modules
- **Code review**: Ensuring API consistency and naming conventions

### Usage

```bash
cd admin_utils
python library_structure.py
```

This will:
1. Scan `../src/kinopulse` directory
2. Parse all `.py` files using AST
3. Extract class definitions, methods, and functions
4. Write results to `library_structure.txt`

### What Gets Captured

**Classes:**
- Class name and inheritance (e.g., `class Time:` or `class MyClass(BaseClass):`)
- Public methods (no leading underscore)
- Important dunder methods: `__init__`, `__getitem__`, `__setitem__`, `__contains__`, `__eq__`, `__lt__`, `__le__`, `__gt__`, `__ge__`, `__add__`, `__sub__`, `__mul__`, `__repr__`, `__str__`
- Method signatures with parameter names (excluding `self`)

**Functions:**
- Top-level public functions (no leading underscore)
- Function signatures with parameter names

**Example Output:**
```
FILE: core/parameters.py
  class ParameterSpec: validate(param)
  class Parameters: __init__(data, specs, subsystem_params, trainable), __getitem__(key), __setitem__(key, value), __contains__(key), get(key, default), data(), trainable(), fixed(), ...
```

### When to Run

Run `library_structure.py` after:

1. **Adding new methods or classes**
   ```bash
   # Made code changes
   python library_structure.py
   git add library_structure.txt
   git commit -m "Add new feature + update library structure"
   ```

2. **Refactoring APIs**
   ```bash
   # Changed method signatures
   python library_structure.py
   # Verify changes in library_structure.txt
   ```

3. **Adding new modules**
   ```bash
   # Created new files
   python library_structure.py
   # New modules will appear in library_structure.txt
   ```

### Integration with Development Workflow

**For Developers:**
- Run after significant API changes
- Include updated `library_structure.txt` in commits
- Use as quick reference when coding

**For Claude:**
- Check `library_structure.txt` before making API changes
- Reference it when asked about available methods
- Update after implementing new features
- Use to verify API consistency across modules

### Customization

To add more dunder methods to the captured list, edit line 27-29 in `library_structure.py`:

```python
important_dunders = ['__init__', '__getitem__', '__setitem__', '__contains__',
                    '__eq__', '__lt__', '__le__', '__gt__', '__ge__',
                    '__add__', '__sub__', '__mul__', '__repr__', '__str__']
```

### Output Format

The output follows this structure:

```
FILE: relative/path/to/module.py
  class ClassName(BaseClass): method1(arg1, arg2), method2(), property1()
  def function_name(arg1, arg2)

FILE: another/module.py
  ...
```

- Relative paths from `src/kinopulse/`
- Methods shown with simplified signatures (no type hints, no defaults)
- Properties shown as `property_name()` for consistency
- Sorted by file path

### Notes

- **AST-based parsing**: Captures structure without executing code
- **No runtime dependencies**: Only needs Python standard library (`ast`, `os`)
- **Fast**: Processes entire codebase in seconds
- **Safe**: Read-only, doesn't modify source files
- **Portable**: Works on any Python 3.7+ installation

### Future Enhancements

Potential improvements for future versions:

- [ ] Add class properties/attributes detection
- [ ] Include return type hints
- [ ] Add method categorization (public/property/dunder)
- [ ] Generate markdown tables for better readability
- [ ] Add parameter type hints
- [ ] Include docstring summaries
- [ ] Generate per-module summaries
