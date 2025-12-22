"""
Distributed Training Utilities.

Helpers for DDP and multi-GPU training.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional


def setup_distributed() -> tuple:
    """
    Setup distributed training environment.
    
    Returns:
        (rank, local_rank, world_size)
    """
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ.get("RANK", local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    else:
        local_rank = 0
        rank = 0
        world_size = 1
    
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is main (rank 0)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def barrier():
    """Synchronization barrier for distributed training."""
    if dist.is_initialized():
        dist.barrier()


def all_gather_object(obj):
    """Gather objects from all processes."""
    if not dist.is_initialized():
        return [obj]
    
    output = [None for _ in range(get_world_size())]
    dist.all_gather_object(output, obj)
    return output


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)


class DistributedContext:
    """Context manager for distributed training."""
    
    def __init__(self):
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        
    def __enter__(self):
        self.rank, self.local_rank, self.world_size = setup_distributed()
        return self
        
    def __exit__(self, *args):
        cleanup_distributed()
        
    @property
    def is_main(self) -> bool:
        return self.rank == 0
        
    @property 
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")
