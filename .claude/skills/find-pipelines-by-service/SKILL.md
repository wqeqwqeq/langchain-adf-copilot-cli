---
name: find-pipelines-by-service
description: Find all pipelines that use a specific type of linked service (e.g. Snowflake, AzureBlobStorage). Cross-references pipelines, datasets, and linked services.
---
# Find Pipelines by Linked Service Type

Find all pipelines in an ADF instance that use a specific type of linked service, through both direct activity references and indirect dataset references.

## Workflow

### Step 1: Resolve Target

User provides domain + environment.

```
resolve_adf_target(domain, environment)
```

If the user does not specify both, ask for clarification.

### Step 2: List Everything (parallel)

Call all three tools together:

- `adf_linked_service_list()` — names + types
- `adf_pipeline_list()` — saves each pipeline as `pipelines/{name}.json`
- `adf_dataset_list()` — saves all datasets as `datasets.json`

### Step 3: Identify Target Linked Services

From the linked service list, identify which ones match the user's request (e.g. type = `Snowflake` or `SnowflakeV2`).

If unsure about version (e.g. type just says "Snowflake" but user asked specifically for v1 vs v2), call `adf_linked_service_get()` on a few to inspect the full definition and confirm.

Collect the **names** of all matching linked services into a target set.

### Step 4: Cross-Reference with exec_python

Write a Python script that:

1. Loads `datasets.json` — builds a `dataset_name → linked_service_name` lookup
2. Iterates all `pipelines/*.json` files
3. For each pipeline, walks all activities and checks **both paths**:
   - **Direct**: activity itself references a linked service (fields like `linked_service_name`, `resource_linked_service`, `reference_objects`)
   - **Indirect**: activity references a dataset (fields like `dataset`, `inputs`, `outputs`), then looks up that dataset in the `dataset → linked_service` mapping
4. Both paths must be checked — direct alone will miss dataset-based references, indirect alone will miss activity-level references
5. Writes clear logs for each pipeline checked and each match found (for debugging)
6. Outputs result as JSON: `{ "pipeline_name": ["ls_name1", "ls_name2"], ... }`

**Reference code example** (adapt based on actual data structure):

```python
import json, os, glob as g

# --- Config ---
session_dir = os.environ.get("SESSION_DIR", ".")
target_ls_names = {"snowflake_v1_ls", "snowflake_v2_prod"}  # from Step 3

# --- Load datasets ---
with open(os.path.join(session_dir, "datasets.json")) as f:
    datasets = json.load(f)
ds_to_ls = {ds["name"]: ds["linked_service"] for ds in datasets}
print(f"Loaded {len(ds_to_ls)} datasets")

# --- Scan pipelines ---
results = {}
pipeline_files = g.glob(os.path.join(session_dir, "pipelines", "*.json"))

for pf in pipeline_files:
    with open(pf) as f:
        pipeline = json.load(f)

    pipeline_name = pipeline.get("name", os.path.basename(pf))
    matched_ls = set()

    activities = pipeline.get("properties", {}).get("activities", [])

    for activity in activities:
        # --- Direct: activity-level linked service ---
        ls_ref = activity.get("linked_service_name", {})
        if isinstance(ls_ref, dict):
            ref_name = ls_ref.get("reference_name", "")
            if ref_name in target_ls_names:
                matched_ls.add(ref_name)

        # Check typeProperties for resource linked service
        type_props = activity.get("type_properties", {}) or activity.get("typeProperties", {})
        for key in ["resource_linked_service", "linked_service_name"]:
            ref = type_props.get(key, {})
            if isinstance(ref, dict):
                ref_name = ref.get("reference_name", "") or ref.get("referenceName", "")
                if ref_name in target_ls_names:
                    matched_ls.add(ref_name)

        # --- Indirect: dataset references ---
        for ds_field in ["dataset", "inputs", "outputs"]:
            ds_ref = type_props.get(ds_field)
            if ds_ref is None:
                continue
            # Normalize to list
            refs = ds_ref if isinstance(ds_ref, list) else [ds_ref]
            for ref in refs:
                if isinstance(ref, dict):
                    ds_name = ref.get("reference_name", "") or ref.get("referenceName", "")
                    ls_name = ds_to_ls.get(ds_name, "")
                    if ls_name in target_ls_names:
                        matched_ls.add(ls_name)
                        print(f"  [{pipeline_name}] dataset '{ds_name}' -> LS '{ls_name}' (MATCH)")

    if matched_ls:
        results[pipeline_name] = sorted(matched_ls)
        print(f"[MATCH] {pipeline_name}: {sorted(matched_ls)}")
    else:
        print(f"[SKIP] {pipeline_name}: no match")

print(f"\n=== Results: {len(results)} pipelines matched ===")
print(json.dumps(results, indent=2))
```

**Important**: The field names above (e.g. `linked_service_name`, `reference_name`, `type_properties`) are based on the Azure SDK `as_dict()` output. If `exec_python` fails with `KeyError`, read 1-2 pipeline files and `datasets.json` to check the actual JSON keys and adjust accordingly. If `exec_python` returns no elements or not find any match, read 1-2 pipeline and `datasets.json` directly as well

### Step 5: If exec_python Fails or return nothing — Debug

- If error relates to pipeline structure: `read_file("pipelines/<some_pipeline>.json")` to understand actual JSON keys
- If error relates to dataset structure: `read_file("datasets.json")` to check format
- Fix the code based on actual structure

### Step 6: Retry exec_python

Apply the fix and re-run. Maximum 3 attempts total.

### Step 7: Present Results

Output the JSON mapping of `pipeline → [linked_services]`:

```json
{
  "pipeline1": ["snowflake_v1_ls", "snowflake_v2_ls"],
  "pipeline2": ["snowflake_v2_native"]
}
```

Also present as a readable table:

```
| Pipeline | Linked Services |
|---|---|
| pipeline1 | snowflake_v1_ls, snowflake_v2_ls |
| pipeline2 | snowflake_v2_native |
```

## How Linked Services Appear in Pipelines

There are two ways a pipeline can reference a linked service:

1. **Direct (activity-level)**: The activity itself has a `linked_service_name` field, or its `typeProperties` contain `resource_linked_service` or similar fields. Common for Web Activities, Azure Function calls, etc.

2. **Indirect (via dataset)**: The activity references a dataset (through `dataset`, `inputs`, or `outputs`), and the dataset points to a linked service. Common for Copy Activities, Lookup, etc.

Both paths must be checked for complete results.

**Note**: Field names may vary between SDK versions or REST API responses. If the script fails, always verify by reading actual files before retrying.

## Important Notes

- Always call all three list tools in parallel (Step 2) for efficiency
- The `exec_python` script should log every pipeline it checks for debuggability
- If the user asks about a specific version (e.g. "Snowflake v2 only"), use `adf_linked_service_get` to inspect the full definition and distinguish versions
- Dataset count is typically small, so saving all datasets in one file is fine
