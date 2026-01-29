---
name: test-linked-service
description: Test linked service connections — handles IR detection, managed IR activation, and supports testing single, multiple, or all linked services by name or type
---
# Test Linked Service Connections

Test one or more Azure Data Factory linked service connections, automatically handling Integration Runtime dependencies.

## Workflow

### 1. Determine Scope

| User Request | Action |
|---|---|
| Test a specific linked service by name | Go to **Step 2** with that name |
| Test all linked services | `adf_linked_service_list` → collect all names → go to **Step 2** for each |
| Test by type (e.g. "all Snowflake") or several types | `adf_linked_service_list` → filter by matching type(s) → go to **Step 2** for each |

### 2. Get Linked Service Details

Call `adf_linked_service_get(name)` to retrieve the full JSON definition.

From the response, extract the Integration Runtime reference:

- Look for `connectVia.referenceName` in the JSON
- If `connectVia` is **absent or null** → the service uses the **AutoResolve** default IR, skip to **Step 4**
- If `connectVia.referenceName` exists → note the IR name, go to **Step 3**

### 3. Check and Prepare Integration Runtime

Call `adf_integration_runtime_get(ir_name)` to get the IR status.

Determine the IR type and act accordingly:

| IR Type | Action |
|---|---|
| **SelfHosted** | No preparation needed. Verify nodes are online (check `nodes` array in status). If offline, warn the user and skip testing this service. Go to **Step 4** |
| **Managed** | Check if interactive authoring is enabled. If not, call `adf_integration_runtime_enable(ir_name, minutes=10)` and wait for confirmation. Then go to **Step 4** |
| **AutoResolve** / other | No preparation needed. Go to **Step 4** |

**Important**: When testing multiple linked services that share the same Managed IR, enable the IR **once** and reuse it for all services on that IR. Track which IRs have already been enabled to avoid redundant calls.

### 4. Test the Connection

Call `adf_linked_service_test(name)`.

Record the result as **Pass** or **Fail** with the error message if applicable.

### 5. Report Results

After testing all requested services, present a summary table:

```
| Linked Service | Type | Integration Runtime | IR Type | Result |
|---|---|---|---|---|
| my-snowflake | SnowflakeV2 | ir-managed-01 | Managed | Pass |
| my-blob | AzureBlobStorage | (AutoResolve) | - | Pass |
| my-sql | SqlServer | ir-selfhosted | SelfHosted | Fail: connection refused |
```

If any tests failed, provide actionable suggestions:
- **SelfHosted IR offline**: "Check that the self-hosted IR node is running"
- **Authentication error**: "Verify credentials or Key Vault secret"
- **Network error**: "Check firewall rules and private endpoint configuration"
- **Timeout**: "The service may be unreachable from the IR network; check NSG/firewall"

## Important Notes

- Always call `adf_linked_service_get` before testing — never guess the IR configuration
- For bulk testing, group services by IR to minimize `adf_integration_runtime_enable` calls
- Managed IR interactive authoring has a startup time; enable it early and test other non-managed-IR services while waiting if possible
- If `adf_linked_service_list` or `adf_linked_service_get` fails, report the error and continue with remaining services
