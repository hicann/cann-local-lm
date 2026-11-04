# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Generate the checked-in manifest schema from its typed Python contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "examples/manifest_schema/manifest-v1.schema.json"


def manifest_schema() -> dict[str, Any]:
    from executorch_npu.artifacts.manifest import Manifest

    schema = Manifest.model_json_schema(mode="serialization")
    # The wire format always emits the protocol defaults and omits None fields.
    schema["required"] = list(dict.fromkeys(["runtime_protocol", *schema["required"]]))
    protocol = schema["$defs"]["RuntimeProtocol"]
    protocol["required"] = list(protocol["properties"])
    for definition in schema["$defs"].values():
        for field in definition.get("properties", {}).values():
            alternatives = field.get("anyOf", [])
            if {"type": "null"} in alternatives:
                non_null = [item for item in alternatives if item != {"type": "null"}]
                if len(non_null) == 1:
                    del field["anyOf"]
                    field.pop("default", None)
                    field.update(non_null[0])
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://ascend.example/executorch_npu/manifest-v1.schema.json",
        **schema,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Fail if the saved schema is stale"
    )
    args = parser.parse_args()
    schema = manifest_schema()
    if args.check:
        if not SCHEMA_PATH.is_file() or json.loads(SCHEMA_PATH.read_text()) != schema:
            print(
                "Manifest schema is stale; run python scripts/generate_manifest_schema.py",
                file=sys.stderr,
            )
            return 1
        print("Manifest schema is current")
        return 0
    SCHEMA_PATH.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(SCHEMA_PATH)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "python"))
    raise SystemExit(main())
