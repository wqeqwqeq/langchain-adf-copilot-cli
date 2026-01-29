"""
ADF 工具定义

提供 Azure Data Factory 的 Pipeline、Linked Service、Integration Runtime 操作。
所有工具都将数据写入 workspace/ 目录，避免将大量 JSON 放入上下文。
"""

import json
import functools

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

    Saves each pipeline as an individual file to pipelines/{name}.json.
    Use read_file to explore individual pipeline definitions.
    Results are cached for the session.

    Returns:
        List of pipeline names, with individual files saved to pipelines/
    """
    try:
        cache = runtime.context._cache
        if "pipelines" in cache:
            names, pipelines_dir = cache["pipelines"]
        else:
            client = _get_adf_client(runtime)
            session_dir = runtime.context.session_dir
            pipelines_dir = session_dir / "pipelines"
            pipelines_dir.mkdir(parents=True, exist_ok=True)

            names = []
            for p in client.list_pipelines():
                data = p.as_dict()
                name = data.get("name", "unknown")
                (pipelines_dir / f"{name}.json").write_text(
                    json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                names.append(name)

            cache["pipelines"] = (names, pipelines_dir)

        return f"""[OK]

Found {len(names)} pipelines. Each saved to {pipelines_dir}/{{name}}.json

{chr(10).join(f"  - {name}" for name in names[:50])}
{"  ... and more" if len(names) > 50 else ""}

Use read_file("pipelines/<name>.json") to explore a specific pipeline and understand the json structure.
Write code and use exec_python to complete complex analysis tasks if needed. 
"""

    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
@require_adf_config
def adf_pipeline_get(name: str, runtime: ToolRuntime[ADFAgentContext]) -> str:
    """
    Get full definition of a specific pipeline.

    Returns the complete JSON definition directly.
    Use this when the user explicitly asks about a specific pipeline.
    For bulk exploration, use adf_pipeline_list + read_file instead.

    Args:
        name: Name of the pipeline to retrieve

    Returns:
        Full pipeline definition as JSON
    """
    try:
        client = _get_adf_client(runtime)
        pipeline_data = client.get_pipeline(name)
        return f"[OK]\n\n{json.dumps(pipeline_data, indent=2, ensure_ascii=False)}"

    except Exception as e:
        return f"[FAILED] {str(e)}"


# === Linked Service 工具 ===

@tool
@require_adf_config
def adf_linked_service_list(runtime: ToolRuntime[ADFAgentContext]) -> str:
    """
    List all linked services in the Azure Data Factory.

    Returns a lightweight summary of all linked services (name + type).
    Use adf_linked_service_get to retrieve full details of a specific service.
    Results are cached for the session.

    Returns:
        List of linked service names with their types
    """
    try:
        cache = runtime.context._cache
        if "linked_services" not in cache:
            client = _get_adf_client(runtime)
            cache["linked_services"] = client.list_linked_services()

        services = cache["linked_services"]
        summary_lines = [f"  - {s['name']} ({s['type']})" for s in services]

        return f"""[OK]

Found {len(services)} linked services.

{chr(10).join(summary_lines[:50])}
{"  ... and more" if len(services) > 50 else ""}

Use adf_linked_service_get(name) to get full details of a specific service.
"""

    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
@require_adf_config
def adf_linked_service_get(name: str, runtime: ToolRuntime[ADFAgentContext]) -> str:
    """
    Get full definition of a specific linked service.

    Returns the complete JSON definition directly.

    Args:
        name: Name of the linked service to retrieve

    Returns:
        Full linked service definition as JSON
    """
    try:
        client = _get_adf_client(runtime)
        service = client.get_linked_service(name)
        return f"[OK]\n\n{json.dumps(service, indent=2, ensure_ascii=False)}"

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

    Returns a lightweight summary of all IRs (name + type).
    Use adf_integration_runtime_get to retrieve full status of a specific IR.
    Results are cached for the session.

    Returns:
        List of IR names with types
    """
    try:
        cache = runtime.context._cache
        if "integration_runtimes" not in cache:
            client = _get_adf_client(runtime)
            cache["integration_runtimes"] = client.list_integration_runtimes()

        irs = cache["integration_runtimes"]
        summary_lines = [f"  - {ir['name']} ({ir['type']})" for ir in irs]

        return f"""[OK]

Found {len(irs)} Integration Runtimes.

{chr(10).join(summary_lines)}

Use adf_integration_runtime_get(name) to get full status of a specific IR.
"""

    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
@require_adf_config
def adf_integration_runtime_get(name: str, runtime: ToolRuntime[ADFAgentContext]) -> str:
    """
    Get full status of a specific Integration Runtime.

    Returns the complete JSON status directly.

    Args:
        name: Name of the Integration Runtime

    Returns:
        Full IR status as JSON
    """
    try:
        client = _get_adf_client(runtime)
        status = client.get_integration_runtime_status(name)
        return f"[OK]\n\n{json.dumps(status, indent=2, ensure_ascii=False)}"

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
