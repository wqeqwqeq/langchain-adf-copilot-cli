"""
Azure Data Factory 客户端

简化的 ADF 操作封装，使用 DefaultAzureCredential 认证。
不依赖外部 azure_tools 包。
"""

import time
import requests
from typing import List, Dict, Union, Optional

from azure.identity import DefaultAzureCredential
from azure.mgmt.datafactory import DataFactoryManagementClient


class ADFClient:
    """
    Azure Data Factory 客户端

    使用 DefaultAzureCredential 进行认证，封装常用的 ADF 操作。
    """

    def __init__(
        self,
        resource_group: str,
        factory_name: str,
        subscription_id: Optional[str] = None,
        credential: Optional[DefaultAzureCredential] = None,
    ):
        """
        初始化 ADF 客户端

        Args:
            resource_group: Azure 资源组名称
            factory_name: ADF 工厂名称
            subscription_id: 订阅 ID（可选，会自动获取）
            credential: Azure 凭据（可选，会自动创建 DefaultAzureCredential）
        """
        self.resource_group = resource_group
        self.factory_name = factory_name
        self.credential = credential or DefaultAzureCredential()

        # 获取 subscription_id
        if subscription_id:
            self.subscription_id = subscription_id
        else:
            self.subscription_id = self._get_subscription_id()

        # 创建 DataFactory 客户端
        self.client = DataFactoryManagementClient(
            credential=self.credential,
            subscription_id=self.subscription_id,
        )

        # 缓存 token
        self._token = None

    # === Pipeline 操作 ===

    def list_pipelines(self) -> List[Dict]:
        """
        列出所有 Pipelines

        Returns:
            Pipeline 字典列表
        """
        pipelines = self.client.pipelines.list_by_factory(
            resource_group_name=self.resource_group,
            factory_name=self.factory_name,
        )
        return [p.as_dict() for p in pipelines]

    def get_pipeline(self, name: str) -> Dict:
        """
        获取 Pipeline 详情

        Args:
            name: Pipeline 名称

        Returns:
            Pipeline 定义字典
        """
        pipeline = self.client.pipelines.get(
            resource_group_name=self.resource_group,
            factory_name=self.factory_name,
            pipeline_name=name,
        )
        return pipeline.as_dict()

    # === 内部方法 ===

    def _get_subscription_id(self) -> str:
        """获取订阅 ID（优先从环境变量，否则从 Azure CLI 默认订阅）"""
        import os
        import subprocess
        import json

        # 1. 优先从环境变量读取
        sub_id = os.getenv("AZURE_SUBSCRIPTION_ID") or os.getenv("ADF_SUBSCRIPTION_ID")
        if sub_id:
            return sub_id

        # 2. 从 Azure CLI 获取默认订阅
        try:
            result = subprocess.run(
                ["az", "account", "show", "--query", "id", "-o", "tsv"],
                capture_output=True,
                text=True,
                check=True,
            )
            sub_id = result.stdout.strip()
            if sub_id:
                return sub_id
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # 3. 回退到 SDK 方式（保持向后兼容）
        from azure.mgmt.resource import SubscriptionClient

        sub_client = SubscriptionClient(self.credential)
        for sub in sub_client.subscriptions.list():
            return sub.subscription_id
        raise ValueError("No Azure subscription found")

    def _get_token(self) -> str:
        """获取 Bearer Token（用于 REST API 调用）"""
        token = self.credential.get_token("https://management.azure.com/.default")
        return token.token

    # === Linked Service 操作 ===

    def list_linked_services(
        self, filter_by_type: Union[str, List[str], None] = None
    ) -> List[Dict]:
        """
        列出所有 Linked Services

        Args:
            filter_by_type: 按类型过滤（如 "Snowflake", "AzureBlobStorage"）

        Returns:
            Linked Service 字典列表
        """
        linked_services = self.client.linked_services.list_by_factory(
            resource_group_name=self.resource_group,
            factory_name=self.factory_name,
        )

        services_list = []
        for service in linked_services:
            service_dict = service.as_dict()

            if filter_by_type:
                service_type = service_dict.get("properties", {}).get("type")
                if isinstance(filter_by_type, str):
                    if service_type == filter_by_type:
                        services_list.append(service_dict)
                elif isinstance(filter_by_type, list):
                    if service_type in filter_by_type:
                        services_list.append(service_dict)
            else:
                services_list.append(service_dict)

        return services_list

    def get_linked_service(self, name: str) -> Dict:
        """
        获取 Linked Service 详情

        Args:
            name: Linked Service 名称

        Returns:
            Linked Service 定义字典
        """
        # 使用 REST API 获取完整详情（包括 typeProperties）
        api_url = (
            f"https://management.azure.com/subscriptions/{self.subscription_id}"
            f"/resourcegroups/{self.resource_group}"
            f"/providers/Microsoft.DataFactory/factories/{self.factory_name}"
            f"/linkedservices/{name}?api-version=2018-06-01"
        )

        headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
        }

        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.json()

    def test_linked_service(self, name: str) -> Dict:
        """
        测试 Linked Service 连接

        Args:
            name: Linked Service 名称

        Returns:
            测试结果字典，包含 succeeded 字段
        """
        # 获取 linked service 详情
        linked_service = self.get_linked_service(name)

        # 构建请求体
        body = {"linkedService": linked_service}

        # 调用测试 API
        api_url = (
            f"https://management.azure.com/subscriptions/{self.subscription_id}"
            f"/resourcegroups/{self.resource_group}"
            f"/providers/Microsoft.DataFactory/factories/{self.factory_name}"
            f"/testConnectivity?api-version=2018-06-01"
        )

        headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
        }

        response = requests.post(api_url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()

    # === Integration Runtime 操作 ===

    def list_integration_runtimes(self) -> List[Dict]:
        """
        列出所有 Integration Runtimes

        Returns:
            IR 字典列表
        """
        irs = self.client.integration_runtimes.list_by_factory(
            resource_group_name=self.resource_group,
            factory_name=self.factory_name,
        )
        return [ir.as_dict() for ir in irs]

    def get_integration_runtime_status(self, name: str) -> Dict:
        """
        获取 Integration Runtime 状态

        Args:
            name: Integration Runtime 名称

        Returns:
            IR 状态字典
        """
        api_url = (
            f"https://management.azure.com/subscriptions/{self.subscription_id}"
            f"/resourcegroups/{self.resource_group}"
            f"/providers/Microsoft.DataFactory/factories/{self.factory_name}"
            f"/integrationruntimes/{name}/getStatus?api-version=2018-06-01"
        )

        headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
        }

        response = requests.post(api_url, headers=headers)
        response.raise_for_status()
        return response.json()

    def get_integration_runtime_type(self, name: str) -> str:
        """
        获取 Integration Runtime 类型

        Args:
            name: Integration Runtime 名称

        Returns:
            IR 类型（如 "Managed", "SelfHosted"）
        """
        status = self.get_integration_runtime_status(name)
        ir_type = status.get("properties", {}).get("type")
        if not ir_type:
            raise ValueError(f"Integration Runtime type not found for {name}")
        return ir_type

    def is_interactive_authoring_enabled(self, name: str) -> bool:
        """
        检查 Interactive Authoring 是否已启用

        Args:
            name: Integration Runtime 名称

        Returns:
            True 如果已启用
        """
        status = self.get_integration_runtime_status(name)
        interactive_status = (
            status.get("properties", {})
            .get("typeProperties", {})
            .get("interactiveQuery", {})
            .get("status")
        )
        return interactive_status == "Enabled"

    def enable_interactive_authoring(self, name: str, minutes: int = 10) -> None:
        """
        启用 Interactive Authoring

        Args:
            name: Integration Runtime 名称
            minutes: 持续时间（分钟）

        Raises:
            ValueError: 如果 IR 类型不是 Managed
        """
        # 检查 IR 类型
        ir_type = self.get_integration_runtime_type(name)
        if ir_type != "Managed":
            raise ValueError(
                f"Interactive authoring only supported for Managed IR. "
                f"Current type: {ir_type}"
            )

        # 检查是否已启用
        if self.is_interactive_authoring_enabled(name):
            return  # 已启用，无需操作

        # 调用启用 API
        api_url = (
            f"https://management.azure.com/subscriptions/{self.subscription_id}"
            f"/resourcegroups/{self.resource_group}"
            f"/providers/Microsoft.DataFactory/factories/{self.factory_name}"
            f"/integrationruntimes/{name}/enableInteractiveQuery?api-version=2018-06-01"
        )

        headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
        }

        body = {"autoTerminationMinutes": minutes}

        response = requests.post(api_url, headers=headers, json=body)
        response.raise_for_status()

        # 等待启用完成
        max_wait = 180  # 最多等待 3 分钟
        waited = 0
        while waited < max_wait:
            if self.is_interactive_authoring_enabled(name):
                return
            time.sleep(10)
            waited += 10

        raise TimeoutError(
            f"Interactive authoring not enabled after {max_wait} seconds"
        )
