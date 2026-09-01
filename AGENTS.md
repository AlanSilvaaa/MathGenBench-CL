# MathGenBench-CL

## Objective

To create a Curriculum-Aligned Benchmark for Math Word Problem Generation of Chilean Grades 1 to 6 called MathGenBench-CL.

## Tech

- Python
- `uv` as the package manager

## Considerations

- Dont use standar `python`, use `uv` instead. e.g. to add a library, use `uv add <library>`. To run a file, use `uv run <file>`. To run tests, use `uv test`.

## Creating a new release

When creating a new release, follow these steps:

1. Run tests if they exist to ensure everything is working as expected (lint, unit test, etc.). If everything works fine, proceed to the next step.
2. Check what is the latest version of the project by looking at the `package.json` file.
3. Update the version number in the `package.json` file according to semantic versioning (https://semver.org/).
4. Use `gh` CLI to create a new release on GitHub. The description should include the changes made in this release, and any relevant information for users or developers.
