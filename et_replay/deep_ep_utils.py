import os
from typing import Dict

from et_replay.execution_trace import Node
import torch.distributed as dist

try:
    import deep_ep
    import hybrid_ep_cpp

    has_deep_ep = True
    has_hybrid_ep_cpp = True
except ImportError:
    has_deep_ep = False
    has_hybrid_ep_cpp = False

def to_hybrid_ep_dtype(dtype: str):
    if dtype == "uint16_t":
        return hybrid_ep_cpp.APP_TOKEN_DATA_TYPE.UINT16
    elif dtype == "uint8_t":
        return hybrid_ep_cpp.APP_TOKEN_DATA_TYPE.UINT8
    else:
        raise ValueError("Unsupported dtype: %s" % dtype)

def init_hybrid_ep_buffer(node: Node, pg_groups: Dict[int, dist.ProcessGroup]):
    assert node.name == "HybridEPBuffer::__init__", "Node is not a HybridEPBuffer::__init__ node"

    pg_id = node.inputs[0]

    config = hybrid_ep_cpp.BufferConfig()
    config.hidden_dim = node.inputs[1][0]
    config.max_num_of_tokens_per_rank = node.inputs[1][1]
    config.num_of_experts_per_rank = node.inputs[1][2]
    config.num_of_ranks_per_node = node.inputs[1][3]
    config.num_of_nodes = node.inputs[1][4]
    config.token_data_type = to_hybrid_ep_dtype(node.inputs[1][5])
    config.num_of_blocks_preprocessing_api = node.inputs[1][6]
    config.num_of_blocks_dispatch_api = node.inputs[1][7]
    config.num_of_blocks_combine_api = node.inputs[1][8]

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
        load_cached_kernels = load_cached_kernels,
        use_shared_buffer = use_shared_buffer,
        enable_custom_allgather = enable_custom_allgather
    )

def get_hybrid_ep_config_instance(node: Node):
    """Build HybridEpConfigInstance from trace node. Tuple order matches config.cuh to_ivalue_tuple() (23 elements)."""
    
    assert node.name.startswith("HybridEPBuffer::"), "Node is not a HybridEPBuffer:: node"
    config = hybrid_ep_cpp.HybridEpConfigInstance()

    # Hybrid-ep Config
    config.hidden_dim = node.inputs[0][0]
    config.max_num_of_tokens_per_rank = node.inputs[0][1]
    config.num_of_experts_per_rank = node.inputs[0][2]
    config.num_of_ranks_per_node = node.inputs[0][3]
    config.num_of_nodes = node.inputs[0][4]

    # Metadata-preprocessing API Config
    config.num_of_threads_per_block_preprocessing_api = node.inputs[0][5]
    config.num_of_blocks_preprocessing_api = node.inputs[0][6]
    config.num_of_blocks_permute_api = node.inputs[0][7]

    # Dispatch API Config
    config.token_data_type = to_hybrid_ep_dtype(node.inputs[0][8])
    config.num_of_stages_dispatch_api = node.inputs[0][9]
    config.num_of_in_flight_s2g_dispatch_api = node.inputs[0][10]
    config.num_of_tokens_per_chunk_dispatch_api = node.inputs[0][11]
    config.num_of_blocks_dispatch_api = node.inputs[0][12]
    config.forward_dispatch_api = bool(node.inputs[0][13])
    config.device_side_sync_dispatch_api = bool(node.inputs[0][14])

    # Combine API Config
    config.num_of_stages_g2s_combine_api = node.inputs[0][15]
    config.num_of_stages_s2g_combine_api = node.inputs[0][16]
    config.num_of_tokens_per_chunk_combine_api = node.inputs[0][17]
    config.num_of_tokens_per_group_combine_api = node.inputs[0][18]
    config.num_of_blocks_combine_api = node.inputs[0][19]
    config.num_of_additional_in_flight_s2g_combine_api = node.inputs[0][20]
    config.backward_combine_api = bool(node.inputs[0][21])
    config.device_side_sync_combine_api = bool(node.inputs[0][22])

    return config

def build_hybrid_ep_func(hybrid_ep_buffer: hybrid_ep_cpp.HybridEPBuffer, node: Node):
    assert node.name.startswith("HybridEPBuffer::"), "Node is not a HybridEPBuffer:: node"

    if node.name == "HybridEPBuffer::update_buffer":
        def update_buffer(*args):
            return hybrid_ep_buffer.update_buffer(config=args[0])
        return update_buffer, 1
    elif node.name == "HybridEPBuffer::metadata_preprocessing":
        def metadata_preprocessing(*args):
            return hybrid_ep_buffer.metadata_preprocessing(
                config=args[0],
                routing_map=args[1],
                num_of_tokens_per_rank=args[2],
                non_blocking=args[3],
            )
        return metadata_preprocessing, 5
    elif node.name == "HybridEPBuffer::combine":
        def combine(*args):
            # C++ binding uses py::kw_only(); pass as keywords.
            return hybrid_ep_buffer.combine(
                config=args[0],
                hidden=args[1],
                probs=args[2],
                sparse_to_dense_map=args[3],
                rdma_to_attn_map=args[4],
                attn_to_rdma_map=args[5],
                num_of_tokens_per_rank=args[6],
                with_probs=args[7],
            )
        return combine, 2
    elif node.name == "HybridEPBuffer::dispatch":
        def dispatch(*args):
            return hybrid_ep_buffer.dispatch(
                config=args[0],
                hidden=args[1],
                probs=args[2],
                scaling_factor=args[3],
                sparse_to_dense_map=args[4],
                rdma_to_attn_map=args[5],
                attn_to_rdma_map=args[6],
                num_dispatched_tokens_tensor=args[7],
                num_dispatched_tokens=args[8],
                num_of_tokens_per_rank=args[9],
                with_probs=args[10],
            )
        return dispatch, 3
    elif node.name == "HybridEPBuffer::dispatch_with_permute":
        def dispatch_with_permute(*args):
            kwargs = dict(
                config=args[0],
                hidden=args[1],
                probs=args[2],
                scaling_factor=args[3],
                sparse_to_dense_map=args[4],
                rdma_to_attn_map=args[5],
                attn_to_rdma_map=args[6],
                num_dispatched_tokens_tensor= None if args[7] is None else args[7].pin_memory(),
                local_expert_routing_map=args[8],
                row_id_map=args[9],
                num_permuted_tokens=args[10],
                num_of_tokens_per_rank=args[11],
                pad_multiple=args[12],
                non_blocking=args[13],
                with_probs=args[14],
            )
            return hybrid_ep_buffer.dispatch_with_permute(**kwargs)
        return dispatch_with_permute, 6
    elif node.name == "HybridEPBuffer::combine_with_unpermute":
        def combine_with_unpermute(*args):
            return hybrid_ep_buffer.combine_with_unpermute(
                config=args[0],
                hidden=args[1],
                probs=args[2],
                sparse_to_dense_map=args[3],
                rdma_to_attn_map=args[4],
                attn_to_rdma_map=args[5],
                num_dispatched_tokens_tensor=None if args[6] is None else args[6].pin_memory(),
                row_id_map=args[7],
                num_of_tokens_per_rank=args[8],
                pad_multiple=args[9],
                with_probs=args[10],
            )
     
        return combine_with_unpermute, 2
    else:
        raise ValueError("hybrid_ep_cpp: unsupported node name %s" % node.name) 