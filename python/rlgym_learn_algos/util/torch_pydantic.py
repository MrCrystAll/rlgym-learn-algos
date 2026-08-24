from typing import Annotated, Any

import torch
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

_supported_dtypes = {
    str(v)[6:] for v in torch.__dict__.values() if isinstance(v, torch.dtype)
}
dtype_str_regex = "|".join(f"({v})" for v in _supported_dtypes)

_supported_devices = [
    "cpu",
    "cuda",
    "ipu",
    "xpu",
    "mkldnn",
    "opengl",
    "opencl",
    "ideep",
    "hip",
    "ve",
    "fpga",
    "maia",
    "xla",
    "lazy",
    "vulkan",
    "mps",
    "meta",
    "hpu",
    "mtia",
    "privateuseone",
]
device_str_regex = "(" + "|".join(f"({v})" for v in _supported_devices) + ")(:\d+)?"

# Created using the example here: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
class _TorchDtypePydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        from_str_schema = core_schema.chain_schema(
            [
                core_schema.str_schema(pattern=dtype_str_regex),
                core_schema.no_info_plain_validator_function(
                    lambda v: getattr(torch, v)
                ),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_str_schema,
            python_schema=core_schema.union_schema(
                [
                    # check if it's an instance first before doing any further work
                    core_schema.is_instance_schema(torch.dtype),
                    from_str_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: str(v)[6:],
                return_schema=core_schema.literal_schema(list(_supported_dtypes)),
            ),
        )


class _TorchDevicePydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        from_str_schema = core_schema.chain_schema(
            [
                core_schema.str_schema(pattern=device_str_regex),
                core_schema.no_info_plain_validator_function(lambda v: torch.device(v)),
            ]
        )
        from_int_schema = core_schema.chain_schema(
            [
                core_schema.int_schema(ge=0),
                core_schema.no_info_plain_validator_function(lambda v: torch.device(v)),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_str_schema,
            python_schema=core_schema.union_schema(
                [
                    # check if it's an instance first before doing any further work
                    core_schema.is_instance_schema(torch.device),
                    from_str_schema,
                    from_int_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: str(v)[6:],
                return_schema=core_schema.literal_schema(list(_supported_dtypes)),
            ),
        )


PydanticTorchDtype = Annotated[torch.dtype, _TorchDtypePydanticAnnotation]
PydanticTorchDevice = Annotated[torch.device, _TorchDevicePydanticAnnotation]
