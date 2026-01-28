"""
ADF Agent 工具集

导出所有可用工具。
"""

from .general_tools import (
    read_file,
    write_file,
    glob,
    grep,
    list_dir,
    exec_python,
    GENERAL_TOOLS,
)

from .adf_tools import (
    adf_pipeline_list,
    adf_pipeline_get,
    adf_linked_service_list,
    adf_linked_service_get,
    adf_linked_service_test,
    adf_integration_runtime_list,
    adf_integration_runtime_get,
    adf_integration_runtime_enable,
    ADF_TOOLS,
)

from .azure_adf_client import ADFClient

# 所有工具列表
ALL_TOOLS = [
    # General tools
    read_file,
    # write_file,
    glob,
    grep,
    list_dir,
    exec_python,
    # ADF tools
    adf_pipeline_list,
    adf_pipeline_get,
    adf_linked_service_list,
    adf_linked_service_get,
    adf_linked_service_test,
    adf_integration_runtime_list,
    adf_integration_runtime_get,
    adf_integration_runtime_enable,
]

__all__ = [
    # General tools
    "read_file",
    # "write_file",
    "glob",
    "grep",
    "list_dir",
    "exec_python",
    "GENERAL_TOOLS",
    # ADF tools
    "adf_pipeline_list",
    "adf_pipeline_get",
    "adf_linked_service_list",
    "adf_linked_service_get",
    "adf_linked_service_test",
    "adf_integration_runtime_list",
    "adf_integration_runtime_get",
    "adf_integration_runtime_enable",
    "ADF_TOOLS",
    # Azure client
    "ADFClient",
    # All tools
    "ALL_TOOLS",
]
