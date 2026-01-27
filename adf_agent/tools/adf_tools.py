"""
ADF 工具定义

提供 Azure Data Factory 的 Pipeline、Linked Service、Integration Runtime 操作。
所有工具都将数据写入 workspace/ 目录，避免将大量 JSON 放入上下文。
"""

import json
import functools
from typing import Optional

from langchain.tools import tool, ToolRuntime

from ..context import ADFAgentContext
from .azure_adf_client import ADFClient


def require_adf_config(func):
    """装饰器：检查 ADF 配置是否完整"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # runtime 是最后一个参数
        runtime = kwargs.get('runtime') or args[-1]
        config = runtime.context.adf_config

        if not config.is_configured():
            missing = config.missing_fields()
            return f"""[FAILED] ADF configuration incomplete.

Missing required fields: {', '.join(missing)}

Please ask the user to provide:
- resource_group: The Azure resource group name containing the ADF
- factory_name: The Azure Data Factory name

The user needs to set environment variables and restart:
  export ADF_RESOURCE_GROUP=<resource-group-name>
  export ADF_FACTORY_NAME=<factory-name>
"""
        return func(*args, **kwargs)
    return wrapper


def _get_adf_client(runtime: ToolRuntime[ADFAgentContext]) -> ADFClient:
    """获取 ADF 客户端实例"""
    config = runtime.context.adf_config
    credential = runtime.context.credential

    return ADFClient(
        resource_group=config.resource_group,
        factory_name=config.factory_name,
        subscription_id=config.subscription_id,
        credential=credential,
    )


# === Pipeline 工具 ===

@tool
@require_adf_config
def adf_pipeline_list(runtime: ToolRuntime[ADFAgentContext]) -> str:
    """
    List all pipelines in the Azure Data Factory.

    Returns pipeline names and saves full details to workspace/pipelines.json.
    Use exec_python to analyze the saved JSON data.

    Returns:
        List of pipeline names, with full data saved to workspace/pipelines.json
    """
    try:
        client = _get_adf_client(runtime)

        # 获取所有 pipeline
        pipelines_data = client.list_pipelines()

        # 保存到 workspace
        workspace = runtime.context.workspace
        output_file = workspace / "pipelines.json"
        output_file.write_text(json.dumps(pipelines_data, indent=2, ensure_ascii=False), encoding="utf-8")

        # 返回摘要
        names = [p.get("name", "unknown") for p in pipelines_data]
        return f"""[OK]

Found {len(pipelines_data)} pipelines. Full data saved to workspace/pipelines.json

Pipeline names:
{chr(10).join(f"  - {name}" for name in names[:50])}
{"  ... and more" if len(names) > 50 else ""}

Use exec_python to analyze the data:
  data = load_json("pipelines.json")
  print(json.dumps(data[0], indent=2))  # View first pipeline structure
"""

    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
@require_adf_config
def adf_pipeline_get(name: str, runtime: ToolRuntime[ADFAgentContext]) -> str:
    """
    Get detailed definition of a specific pipeline.

    Saves the pipeline definition to workspace/pipelines/{name}.json.
    Use this to get activities, parameters, and other pipeline details.

    Args:
        name: Name of the pipeline to retrieve

    Returns:
        Pipeline summary, with full definition saved to workspace/pipelines/{name}.json
    """
    try:
        client = _get_adf_client(runtime)

        # 获取 pipeline 定义
        pipeline_data = client.get_pipeline(name)

        # 保存到 workspace/pipelines/
        workspace = runtime.context.workspace
        pipelines_dir = workspace / "pipelines"
        pipelines_dir.mkdir(parents=True, exist_ok=True)
        output_file = pipelines_dir / f"{name}.json"
        output_file.write_text(json.dumps(pipeline_data, indent=2, ensure_ascii=False), encoding="utf-8")

        # 提取摘要信息
        activities = pipeline_data.get("properties", {}).get("activities", [])
        activity_names = [a.get("name", "unknown") for a in activities]
        parameters = list(pipeline_data.get("properties", {}).get("parameters", {}).keys())

        return f"""[OK]

Pipeline: {name}
Activities ({len(activities)}): {', '.join(activity_names[:10])}{"..." if len(activity_names) > 10 else ""}
Parameters: {', '.join(parameters) if parameters else "(none)"}

Full definition saved to workspace/pipelines/{name}.json

Use exec_python to analyze:
  pipeline = load_json("pipelines/{name}.json")
  activities = pipeline["properties"]["activities"]
"""

    except Exception as e:
        return f"[FAILED] {str(e)}"


# === Linked Service 工具 ===

@tool
@require_adf_config
def adf_linked_service_list(filter_type: Optional[str] = None, runtime: ToolRuntime[ADFAgentContext] = None) -> str:
    """
    List all linked services in the Azure Data Factory.

    Optionally filter by service type (e.g., "Snowflake", "AzureBlobStorage").
    Saves full details to workspace/linked_services.json.

    Args:
        filter_type: Optional type to filter by (e.g., "Snowflake", "AzureBlobStorage", "AzureSqlDatabase")

    Returns:
        List of linked service names with types, full data saved to workspace/linked_services.json
    """
    try:
        client = _get_adf_client(runtime)

        # 获取所有 linked services
        services = client.list_linked_services(filter_by_type=filter_type)

        # 保存到 workspace
        workspace = runtime.context.workspace
        output_file = workspace / "linked_services.json"
        output_file.write_text(json.dumps(services, indent=2, ensure_ascii=False), encoding="utf-8")

        # 返回摘要
        type_filter_msg = f" (filtered by type: {filter_type})" if filter_type else ""
        summary_lines = []
        for s in services[:50]:
            name = s.get("name", "unknown")
            svc_type = s.get("properties", {}).get("type", "unknown")
            summary_lines.append(f"  - {name} ({svc_type})")

        return f"""[OK]

Found {len(services)} linked services{type_filter_msg}. Full data saved to workspace/linked_services.json

{chr(10).join(summary_lines)}
{"  ... and more" if len(services) > 50 else ""}

Use exec_python to analyze:
  services = load_json("linked_services.json")
  # Group by type
  from collections import Counter
  types = Counter(s["properties"]["type"] for s in services)
  print(types)
"""

    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
@require_adf_config
def adf_linked_service_get(name: str, runtime: ToolRuntime[ADFAgentContext]) -> str:
    """
    Get detailed information about a specific linked service.

    Saves the linked service definition to workspace/linked_service_{name}.json.

    Args:
        name: Name of the linked service to retrieve

    Returns:
        Linked service summary, with full definition saved to workspace/linked_service_{name}.json
    """
    try:
        client = _get_adf_client(runtime)

        # 获取详情
        service = client.get_linked_service(name)

        # 保存到 workspace
        workspace = runtime.context.workspace
        output_file = workspace / f"linked_service_{name}.json"
        output_file.write_text(json.dumps(service, indent=2, ensure_ascii=False), encoding="utf-8")

        # 提取摘要
        props = service.get("properties", {})
        svc_type = props.get("type", "unknown")
        type_props = props.get("typeProperties", {})

        # 根据类型提取关键信息
        key_info = []
        if svc_type == "Snowflake":
            if "connectionString" in type_props:
                key_info.append("Connection: connectionString configured")
            if "accountIdentifier" in type_props:
                key_info.append(f"Account: {type_props.get('accountIdentifier', 'N/A')}")
        elif svc_type in ("AzureBlobStorage", "AzureBlobFS"):
            if "serviceEndpoint" in type_props:
                key_info.append(f"Endpoint: {type_props.get('serviceEndpoint', 'N/A')}")
        elif svc_type == "AzureSqlDatabase":
            if "connectionString" in type_props:
                key_info.append("Connection: connectionString configured")

        # 检查 connectVia (Integration Runtime)
        connect_via = props.get("connectVia", {})
        if connect_via:
            ir_name = connect_via.get("referenceName", "unknown")
            key_info.append(f"Integration Runtime: {ir_name}")

        return f"""[OK]

Linked Service: {name}
Type: {svc_type}
{chr(10).join(key_info) if key_info else "(no additional info extracted)"}

Full definition saved to workspace/linked_service_{name}.json

Use exec_python to analyze:
  service = load_json("linked_service_{name}.json")
  print(json.dumps(service["properties"], indent=2))
"""

    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
@require_adf_config
def adf_linked_service_test(name: str, runtime: ToolRuntime[ADFAgentContext]) -> str:
    """
    Test the connection of a linked service.

    Note: Some linked services require Integration Runtime to be enabled for testing.
    If the test fails due to IR not being enabled, use adf_integration_runtime_enable first.

    Args:
        name: Name of the linked service to test

    Returns:
        Test result (success or failure with error message)
    """
    try:
        client = _get_adf_client(runtime)

        # 执行连接测试
        result = client.test_linked_service(name)

        if result.get("succeeded"):
            return f"""[OK]

Linked service '{name}' connection test PASSED.

The connection is working correctly.
"""
        else:
            errors = result.get("errors", [])
            error_msg = errors[0].get("message", "Unknown error") if errors else "Unknown error"
            return f"""[FAILED] Linked service '{name}' connection test FAILED.

Error: {error_msg}

Possible causes:
1. Integration Runtime is not enabled (use adf_integration_runtime_enable)
2. Credentials are incorrect
3. Network connectivity issues
4. Target service is unavailable
"""

    except Exception as e:
        error_str = str(e)
        # 检查是否是 IR 相关错误
        if "interactive authoring" in error_str.lower() or "integration runtime" in error_str.lower():
            return f"""[FAILED] {error_str}

This error may be due to Integration Runtime not being enabled.
Try: adf_integration_runtime_enable(name="<ir-name>", minutes=10)
"""
        return f"[FAILED] {str(e)}"


# === Integration Runtime 工具 ===

@tool
@require_adf_config
def adf_integration_runtime_list(runtime: ToolRuntime[ADFAgentContext]) -> str:
    """
    List all Integration Runtimes in the Azure Data Factory.

    Saves full details to workspace/integration_runtimes.json.

    Returns:
        List of IR names with types, full data saved to workspace/integration_runtimes.json
    """
    try:
        client = _get_adf_client(runtime)

        # 获取所有 Integration Runtimes
        irs_data = client.list_integration_runtimes()

        # 保存到 workspace
        workspace = runtime.context.workspace
        output_file = workspace / "integration_runtimes.json"
        output_file.write_text(json.dumps(irs_data, indent=2, ensure_ascii=False), encoding="utf-8")

        # 返回摘要
        summary_lines = []
        for ir in irs_data:
            name = ir.get("name", "unknown")
            ir_type = ir.get("properties", {}).get("type", "unknown")
            summary_lines.append(f"  - {name} ({ir_type})")

        return f"""[OK]

Found {len(irs_data)} Integration Runtimes. Full data saved to workspace/integration_runtimes.json

{chr(10).join(summary_lines)}

Types:
- Managed: Azure-hosted, supports interactive authoring
- SelfHosted: On-premises or VM-hosted

Use adf_integration_runtime_get(name) to check status and interactive authoring.
"""

    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
@require_adf_config
def adf_integration_runtime_get(name: str, runtime: ToolRuntime[ADFAgentContext]) -> str:
    """
    Get detailed status of a specific Integration Runtime.

    Includes interactive authoring status for Managed IRs.
    Saves full status to workspace/ir_{name}_status.json.

    Args:
        name: Name of the Integration Runtime

    Returns:
        IR status summary, with full data saved to workspace/ir_{name}_status.json
    """
    try:
        client = _get_adf_client(runtime)

        # 获取状态
        status = client.get_integration_runtime_status(name)

        # 保存到 workspace
        workspace = runtime.context.workspace
        output_file = workspace / f"ir_{name}_status.json"
        output_file.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

        # 提取关键信息
        props = status.get("properties", {})
        ir_type = props.get("type", "unknown")
        type_props = props.get("typeProperties", {})

        # 检查 interactive authoring 状态（仅 Managed）
        interactive_status = "N/A"
        if ir_type == "Managed":
            interactive_query = type_props.get("interactiveQuery", {})
            interactive_status = interactive_query.get("status", "Disabled")

        return f"""[OK]

Integration Runtime: {name}
Type: {ir_type}
Interactive Authoring: {interactive_status}

Full status saved to workspace/ir_{name}_status.json

{"Note: Interactive authoring is required for connection testing. Use adf_integration_runtime_enable to enable it." if interactive_status == "Disabled" else ""}
"""

    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
@require_adf_config
def adf_integration_runtime_enable(name: str, minutes: int = 10, runtime: ToolRuntime[ADFAgentContext] = None) -> str:
    """
    Enable interactive authoring for a Managed Integration Runtime.

    Interactive authoring is required for:
    - Testing linked service connections
    - Previewing data in datasets
    - Running debug pipeline runs

    This may take 1-2 minutes to complete.

    Args:
        name: Name of the Integration Runtime
        minutes: Duration to keep interactive authoring enabled (default: 10, max: 120)

    Returns:
        Success message when interactive authoring is enabled
    """
    try:
        client = _get_adf_client(runtime)

        # 检查 IR 类型
        ir_type = client.get_integration_runtime_type(name)
        if ir_type != "Managed":
            return f"""[FAILED] Interactive authoring is only supported for Managed Integration Runtimes.

Integration Runtime '{name}' is of type '{ir_type}'.
Only 'Managed' type supports interactive authoring.
"""

        # 检查是否已启用
        if client.is_interactive_authoring_enabled(name):
            return f"""[OK]

Interactive authoring is already enabled for Integration Runtime '{name}'.
You can proceed with connection testing.
"""

        # 启用 interactive authoring
        client.enable_interactive_authoring(name, minutes=minutes)

        return f"""[OK]

Interactive authoring enabled for Integration Runtime '{name}'.
Duration: {minutes} minutes (auto-terminates after)

You can now:
- Test linked service connections with adf_linked_service_test
- Preview data in datasets
- Run debug pipeline executions
"""

    except TimeoutError as e:
        return f"[FAILED] {str(e)}"
    except ValueError as e:
        return f"[FAILED] {str(e)}"
    except Exception as e:
        return f"[FAILED] {str(e)}"


# 导出所有 ADF 工具
ADF_TOOLS = [
    adf_pipeline_list,
    adf_pipeline_get,
    adf_linked_service_list,
    adf_linked_service_get,
    adf_linked_service_test,
    adf_integration_runtime_list,
    adf_integration_runtime_get,
    adf_integration_runtime_enable,
]
