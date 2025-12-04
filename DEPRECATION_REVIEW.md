# Deprecation Review Summary

## Overview

This document summarizes all deprecated references to `langextract` for extraction that have been identified and updated.

## Status: ✅ Complete

All deprecated references have been marked or updated. `langextract` is now **only used for visualization** (`langextract.visualize()`), not for extraction.

## Changes Made

### Documentation Files Updated

1. **specs/001-schema-corpus-extractor/plan.md**

   - Updated summary to mention PydanticAI instead of langextract
   - Updated dependencies list
   - Updated file structure comments

2. **specs/001-schema-corpus-extractor/research.md**

   - Added "Why not langextract" section explaining technical limitations
   - Updated implementation notes

3. **specs/001-schema-corpus-extractor/data-model.md**

   - Updated PrePromptInteraction and PrePromptInstrumentation docstrings to mark as deprecated

4. **specs/002-structured-extraction/spec.md**

   - Added explanation of why langextract was replaced
   - Clarified that langextract is only for visualization

5. **specs/002-structured-extraction/plan.md**

   - Updated summary and dependencies
   - Updated file structure comments

6. **specs/002-structured-extraction/research.md**

   - Added PydanticAI section with "Why not langextract" explanation
   - Updated OpenAI/Gemini sections to note they're via PydanticAI

7. **specs/002-structured-extraction/quickstart.md**

   - Added explanation of why langextract isn't used

8. **specs/002-structured-extraction/data-model.md**

   - Updated to note models are used by both pipelines

9. **specs/001-schema-corpus-extractor/tasks.md**
   - Updated dependency list to note langextract is visualization-only

### Test Files Updated (with deprecation warnings)

1. **tests/unit/test_extract.py**

   - Added deprecation note in module docstring
   - Added deprecation comment on @patch decorator
   - Marked langextract imports as deprecated

2. **tests/integration/test_end_to_end.py**

   - Added deprecation note in module docstring
   - Added deprecation comment on @patch decorator
   - Marked langextract imports as deprecated

3. **tests/unit/test_edge_cases.py**
   - Added deprecation note in module docstring
   - Added deprecation comment on @patch decorator
   - Marked langextract imports as deprecated

**Note**: These tests still mock `langextract.extract` which no longer exists. They need to be updated to mock PydanticAI Agent instead. This is a separate task.

### Code Files (Already Updated)

1. **src/ctrlf/app/extract.py** - ✅ Uses PydanticAI
2. **src/ctrlf/app/structured_extract.py** - ✅ Uses PydanticAI
3. **src/ctrlf/app/models.py** - ✅ Docstrings marked as deprecated

### Remaining Valid Uses

These references to `langextract` are **valid** and should remain:

- `langextract.visualize()` - Used for HTML visualization
- Dependency in `pyproject.toml` - Required for visualization
- References in visualization-related documentation

## Why langextract Was Replaced

The real reasons (now properly documented):

1. **Requires in-context examples**: langextract requires few-shot examples in the prompt, which adds complexity and token overhead
2. **Cannot condition on schema**: Unlike modern APIs, langextract cannot directly use JSON Schema to constrain outputs - it relies on prompt engineering with examples
3. **Less flexible**: The need for examples makes it harder to adapt to different schemas dynamically

## Next Steps

1. **Update tests** to mock PydanticAI Agent instead of `langextract.extract`
2. **Remove** `langextract` imports from test files once tests are updated
3. **Consider** removing `PrePromptInteraction` and `PrePromptInstrumentation` models if they're no longer needed
