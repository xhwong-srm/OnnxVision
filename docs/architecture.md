# Model workflow architecture

The application selects a model with three independent values:

- `task`: the prediction contract, such as `classification` or `object-detection`;
- `framework`: the implementation provider, such as `timm` or `ultralytics`;
- `model`: the application's canonical model ID, such as `resnet18` or `yolo26n`.

A built-in `FrameworkTaskPlugin` owns one `(framework, task)` pair. It publishes:

- a model catalog that resolves canonical IDs to native IDs;
- only the operation handlers it supports;
- a parameter schema for each operation;
- its dataset requirements and optional dependencies.

The registry performs discovery only. It does not route individual parameters.
Before execution, the service asks the selected operation handler for its schema,
rejects unknown values, resolves defaults, and passes a `Resolved*Request` to the
handler. Run manifests retain both requested and effective values plus the native
model ID and installed dependency versions.

Parameter schemas are composable. Shared training settings form the first layer;
a framework, task, or model layer can add or replace defaults while preserving the
parameter type. The CLI is built from the same final schema, so CLI support and
runtime support cannot drift independently.

## Adding support

To add a model to a static catalog, add a `ModelInfo` entry. Frameworks such as
timm use a dynamic catalog and need no per-model backend class.

To add a new framework/task pair:

1. implement plain operation functions for the supported operations;
2. define each operation's parameter schema and dataset requirement;
3. define a static or dynamic model catalog;
4. register one `FrameworkTaskPlugin` in `backends/registry.py`;
5. test model resolution, parameter rejection/defaults, and one operation boundary.

An integration does not inherit a wide backend interface and does not implement
placeholder operations. Native argument translation remains inside its operation
implementation.
