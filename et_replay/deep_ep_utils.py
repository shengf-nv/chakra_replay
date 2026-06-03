import os
from typing import Dict, List, Optional, Tuple, Union

from et_replay.execution_trace import Node
import torch
import torch.distributed as dist

try:
    import deep_ep
    import hybrid_ep_cpp

    has_deep_ep = True
    has_hybrid_ep_cpp = True
except ImportError:
    has_deep_ep = False
    has_hybrid_ep_cpp = False

# Handle tuple field indices (matches HandleImpl::to_ivalue_tuple in executor.cuh).
_HANDLE_TENSOR_INDICES = (0, 1, 2, 3, 4, 7, 8, 9, 11, 12)
_HANDLE_CONFIG_INDEX = 6

_HYBRID_EP_HANDLE_INPUT_IDX = {
    "HybridEPBuffer::dispatch": 3,
    "HybridEPBuffer::combine": 2,
    "HybridEPBuffer::dispatch_with_permute": 3,
    "HybridEPBuffer::combine_with_unpermute": 2,
}


def to_hybrid_ep_dtype(dtype: str):
    if dtype == "uint16_t":
        return hybrid_ep_cpp.APP_TOKEN_DATA_TYPE.UINT16
    elif dtype == "uint8_t":
        return hybrid_ep_cpp.APP_TOKEN_DATA_TYPE.UINT8
    else:
        raise ValueError("Unsupported dtype: %s" % dtype)


def is_uninitialized_tensor(value) -> bool:
    if value is None or value == "<None>":
        return True
    if isinstance(value, (list, tuple)) and len(value) >= 1 and value[0] == 69:
        return True
    return False


def is_hybrid_ep_config_input(node: Node, idx: int) -> bool:
    return node.name in (
        "HybridEPBuffer::update_buffer",
        "HybridEPBuffer::metadata_preprocessing",
    ) and idx == 0


def is_hybrid_ep_handle_input(node: Node, idx: int) -> bool:
    return _HYBRID_EP_HANDLE_INPUT_IDX.get(node.name) == idx


def get_hybrid_ep_config_instance_from_tuple(config_tuple: Tuple) -> hybrid_ep_cpp.HybridEpConfigInstance:
    """Build HybridEpConfigInstance from tuple. Order matches config.cuh to_ivalue_tuple() (32 elements)."""
    config = hybrid_ep_cpp.HybridEpConfigInstance()

    # Hybrid-ep Config
    config.hidden_dim = config_tuple[0]
    config.max_num_of_tokens_per_rank = config_tuple[1]
    config.num_of_experts_per_rank = config_tuple[2]
    config.num_of_ranks_per_node = config_tuple[3]
    config.num_of_nodes = config_tuple[4]
    config.pad_multiple = config_tuple[5]

    # Metadata-preprocessing API Config
    config.num_of_tokens_per_chunk_preprocessing_api = config_tuple[6]
    config.num_of_threads_per_block_preprocessing_api = config_tuple[7]
    config.num_of_blocks_preprocessing_api = config_tuple[8]
    config.num_of_blocks_permute = config_tuple[9]
    config.num_of_blocks_unpermute = config_tuple[10]

    # Dispatch API Config
    config.token_data_type = to_hybrid_ep_dtype(config_tuple[11])
    config.num_of_stages_dispatch_api = config_tuple[12]
    config.num_of_stages_permute_block_dispatch_api = config_tuple[13]
    config.num_of_in_flight_s2g_dispatch_api = config_tuple[14]
    config.num_of_in_flight_s2g_permute_block_dispatch_api = config_tuple[15]
    config.num_of_additional_in_flight_s2g_dispatch_api = config_tuple[16]
    config.num_of_tokens_per_chunk_dispatch_api = config_tuple[17]
    config.num_of_blocks_dispatch_api = config_tuple[18]
    config.forward_dispatch_api = bool(config_tuple[19])
    config.device_side_sync_dispatch_api = bool(config_tuple[20])

    # Combine API Config
    config.num_of_stages_g2s_combine_api = config_tuple[21]
    config.num_of_stages_s2g_combine_api = config_tuple[22]
    config.num_of_stages_g2s_unpermute_block = config_tuple[23]
    config.num_of_stages_s2g_unpermute_block = config_tuple[24]
    config.num_of_tokens_per_chunk_combine_api = config_tuple[25]
    config.num_of_tokens_per_group_combine_api = config_tuple[26]
    config.num_of_blocks_combine_api = config_tuple[27]
    config.num_of_additional_in_flight_s2g_combine_api = config_tuple[28]
    config.num_of_additional_in_flight_s2g_unpermute_block_combine_api = config_tuple[29]
    config.backward_combine_api = bool(config_tuple[30])
    config.device_side_sync_combine_api = bool(config_tuple[31])

    return config


def get_hybrid_ep_config_instance(node: Node):
    """Build HybridEpConfigInstance from trace node inputs[0]."""
    assert node.name.startswith("HybridEPBuffer::"), "Node is not a HybridEPBuffer:: node"
    return get_hybrid_ep_config_instance_from_tuple(node.inputs[0])


def _split_generic_list_type(type_str: str) -> List[str]:
    """Split GenericList[A, B, ...] into top-level element type strings."""
    if not type_str.startswith("GenericList[") or not type_str.endswith("]"):
        raise ValueError("Expected GenericList[...] type, got %s" % type_str)
    inner = type_str[len("GenericList[") : -1]
    result: List[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(inner):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "," and depth == 0:
            data = inner[start:i].strip()
            if data.startswith("Tensor"):
                result.append(data)
            start = i + 1
    result.append(inner[start:].strip())
    return result


def _resolve_handle_tensor(
    node: Node,
    value,
    type,
    tensor_registry: Dict,
    tensors_mapping: Dict,
    tensor_with_device: bool,
) -> Optional[torch.Tensor]:
    if type == "Tensor(nullptr (uninitialized))":
        return torch.tensor([], dtype=torch.int64)
    t_id = tuple(value[:5]) if tensor_with_device else tuple(value)
    return tensor_registry[tensors_mapping[(node.id, t_id, True)]]


def resolve_hybrid_ep_handle(
    node: Node,
    value: Tuple,
    type: str,
    tensor_registry: Dict,
    tensors_mapping: Dict,
    tensor_with_device: bool = True,
) -> hybrid_ep_cpp.HandleImpl:
    """Resolve traced handle tuple (tensor ids + nested config) into HandleImpl."""
    type = _split_generic_list_type(type)
    resolved = list(value)
    tensor_idx = 0
    for idx in _HANDLE_TENSOR_INDICES:
        resolved[idx] = _resolve_handle_tensor(
            node, value[idx], type[tensor_idx], tensor_registry, tensors_mapping, tensor_with_device
        )
        tensor_idx += 1
    if resolved[3] is not None:
        resolved[3] = resolved[3].pin_memory()
    resolved[_HANDLE_CONFIG_INDEX] = get_hybrid_ep_config_instance_from_tuple(value[_HANDLE_CONFIG_INDEX])
    return get_hybrid_ep_handle_instance(resolved)


def get_hybrid_ep_handle_instance(value: Union[Tuple, hybrid_ep_cpp.HandleImpl]) -> hybrid_ep_cpp.HandleImpl:
    """Build HandleImpl from a resolved handle tuple (13 elements)."""
    if isinstance(value, hybrid_ep_cpp.HandleImpl):
        return value
    
    handle = hybrid_ep_cpp.HandleImpl()
    handle.sparse_to_dense_map = value[0]
    handle.rdma_to_attn_map = value[1]
    handle.attn_to_rdma_map = value[2]
    handle.num_dispatched_tokens_tensor = value[3]
    handle.local_expert_routing_map = value[4]
    handle.num_of_tokens_per_rank = value[5]
    if isinstance(value[6], hybrid_ep_cpp.HybridEpConfigInstance):
        handle.config = value[6]
    else:
        handle.config = get_hybrid_ep_config_instance_from_tuple(value[6])
    handle.tokens_per_expert = value[7]
    handle.padded_tokens_per_expert = value[8]
    handle.overflow_flag = value[9]
    handle.num_permuted_tokens = value[10]
    handle.dense_chunk_layout = value[11]
    handle.dense_to_expert_map = value[12]
    return handle


def init_hybrid_ep_buffer(node: Node, pg_groups: Dict[int, dist.ProcessGroup]):
    assert node.name == "HybridEPBuffer::__init__", "Node is not a HybridEPBuffer::__init__ node"

    pg_id = node.inputs[0]

    config = hybrid_ep_cpp.BufferConfig()
    config_tuple = node.inputs[1]
    config.hidden_dim = config_tuple[0]
    config.max_num_of_tokens_per_rank = config_tuple[1]
    config.num_of_experts_per_rank = config_tuple[2]
    config.num_of_ranks_per_node = config_tuple[3]
    config.num_of_nodes = config_tuple[4]
    config.token_data_type = to_hybrid_ep_dtype(config_tuple[5])
    config.num_of_blocks_preprocessing_api = config_tuple[6]
    config.num_of_blocks_dispatch_api = config_tuple[7]
    config.num_of_blocks_combine_api = config_tuple[8]
    config.num_of_tokens_per_chunk_dispatch_api = config_tuple[9]
    config.num_of_tokens_per_chunk_combine_api = config_tuple[10]
    config.num_of_dispatch_chunks = config_tuple[11]
    config.num_of_combine_chunks = config_tuple[12]

    local_rank = node.inputs[2]
    node_rank = node.inputs[3]
    group_size = node.inputs[4]
    _ = node.inputs[5]
    load_cached_kernels = node.inputs[6]
    use_shared_buffer = node.inputs[7]
    enable_custom_allgather = node.inputs[8]

    return hybrid_ep_cpp.HybridEPBuffer(
        pg_groups[int(pg_id)],
        config,
        local_rank,
        node_rank,
        group_size,
        os.path.dirname(os.path.abspath(deep_ep.__file__)),
        load_cached_kernels=load_cached_kernels,
        use_shared_buffer=use_shared_buffer,
        enable_custom_allgather=enable_custom_allgather,
    )


def _handle_to_output_tuple(handle: hybrid_ep_cpp.HandleImpl) -> Tuple:
    return (
        handle.sparse_to_dense_map,
        handle.rdma_to_attn_map,
        handle.attn_to_rdma_map,
        handle.num_dispatched_tokens_tensor,
        handle.local_expert_routing_map,
        handle.num_of_tokens_per_rank,
        handle.config,
        handle.tokens_per_expert,
        handle.padded_tokens_per_expert,
        handle.overflow_flag,
        handle.num_permuted_tokens,
        handle.dense_chunk_layout,
        handle.dense_to_expert_map,
    )


def build_hybrid_ep_func(hybrid_ep_buffer: hybrid_ep_cpp.HybridEPBuffer, node: Node):
    assert node.name.startswith("HybridEPBuffer::"), "Node is not a HybridEPBuffer:: node"

    if node.name == "HybridEPBuffer::update_buffer":
        def update_buffer(*args):
            return hybrid_ep_buffer.update_buffer(config=args[0])

        return update_buffer, 1
    elif node.name == "HybridEPBuffer::metadata_preprocessing":
        def metadata_preprocessing(*args):
            handle = hybrid_ep_buffer.metadata_preprocessing(
                config=args[0],
                routing_map=args[1],
                num_of_tokens_per_rank=args[2],
                num_permuted_tokens=args[3],
                pad_multiple=args[4],
                enable_permute=args[5],
                fuse_permute_dispatch=args[6],
                non_blocking=args[7],
            )
            return _handle_to_output_tuple(handle)

        return metadata_preprocessing, 13
    elif node.name == "HybridEPBuffer::combine":
        def combine(*args):
            return hybrid_ep_buffer.combine(
                hidden=args[0],
                probs=args[1],
                handle=get_hybrid_ep_handle_instance(args[2]),
                with_probs=args[3],
            )

        return combine, 2
    elif node.name == "HybridEPBuffer::dispatch":
        def dispatch(*args):
            return hybrid_ep_buffer.dispatch(
                hidden=args[0],
                probs=args[1],
                scaling_factor=args[2],
                handle=get_hybrid_ep_handle_instance(args[3]),
                with_probs=args[4],
            )

        return dispatch, 3
    elif node.name == "HybridEPBuffer::dispatch_with_permute":
        def dispatch_with_permute(*args):
            output = hybrid_ep_buffer.dispatch_with_permute(
                hidden=args[0],
                probs=args[1],
                scaling_factor=args[2],
                handle=get_hybrid_ep_handle_instance(args[3]),
                pad_multiple=args[4],
                fuse_permute_dispatch=args[5],
                non_blocking=args[6],
                with_probs=args[7],
            )
            torch.cuda.synchronize(torch.cuda.current_device())
            return output

        return dispatch_with_permute, 3
    elif node.name == "HybridEPBuffer::combine_with_unpermute":
        def combine_with_unpermute(*args):
            return hybrid_ep_buffer.combine_with_unpermute(
                hidden=args[0],
                probs=args[1],
                handle=get_hybrid_ep_handle_instance(args[2]),
                pad_multiple=args[3],
                fuse_unpermute_combine=args[4],
                with_probs=args[5],
            )

        return combine_with_unpermute, 2
    else:
        raise ValueError("hybrid_ep_cpp: unsupported node name %s" % node.name)
