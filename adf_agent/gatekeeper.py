"""
ADF target registry — domain/env → config mapping.
"""

from dataclasses import dataclass


@dataclass
class ADFTargetConfig:
    resource_group: str
    factory_name: str
    subscription_id: str


TargetMap = dict[str, dict[str, ADFTargetConfig]]

ADF_TARGETS: TargetMap = {
    "sales": {
        "dev": ADFTargetConfig("rg-sales-dev", "adf-sales-dev", "00000000-0000-0000-0000-000000000001"),
        "qa": ADFTargetConfig("rg-sales-qa", "adf-sales-qa", "00000000-0000-0000-0000-000000000001"),
        "prod": ADFTargetConfig("rg-sales-prod", "adf-sales-prod", "00000000-0000-0000-0000-000000000001"),
    },
    "hr": {
        "dev": ADFTargetConfig("rg-hr-dev", "adf-hr-dev", "00000000-0000-0000-0000-000000000002"),
        "qa": ADFTargetConfig("rg-hr-qa", "adf-hr-qa", "00000000-0000-0000-0000-000000000002"),
        "prod": ADFTargetConfig("rg-hr-prod", "adf-hr-prod", "00000000-0000-0000-0000-000000000002"),
    },
    "personal": {
        "prod": ADFTargetConfig("adf", "stanley-adf", "ee5f77a1-2e59-4335-8bdf-f7ea476f6523"),
    },
}
