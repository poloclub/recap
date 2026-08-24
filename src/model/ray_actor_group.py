"""Base classes for distributing model actors (policy/reference/critic/reward) across Ray.

Provides the torch-distributed bootstrap shared by all model actors
(:class:`BaseDistributedActor`, :class:`BaseModelActor`) and
:class:`RayActorGroup`, which creates and coordinates a group of such actors
(one per rank) over a Ray placement group and exposes async helpers to
fan out remote calls, including batched, chunked dispatch across actors.
"""

import logging
import os
import socket
from typing import Dict, Type, Union

import ray
import torch
from openrlhf.trainer.ray.utils import ray_noset_visible_devices
from openrlhf.utils.deepspeed import DeepspeedStrategy
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from tqdm import tqdm


class BaseDistributedActor:
    """Sets up torch-distributed environment variables for a single rank of a Ray actor group."""

    def __init__(self, world_size, rank, master_addr, master_port):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set the *_VISIBLE_DEVICES
        # environment variable for each actor, unless
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set, so
        # set local rank to 0 when the flag is not applicable.
        os.environ["LOCAL_RANK"] = (
            str(ray.get_gpu_ids()[0]) if ray_noset_visible_devices() else "0"
        )

    @staticmethod
    def _get_current_node_ip():
        """Return this Ray node's IP address (used as the default master address)."""
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    @staticmethod
    def _get_free_port():
        """Find an available local TCP port (used as the default master port)."""
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        """Return the (address, port) rank 0 uses for torch-distributed rendezvous."""
        return self._master_addr, self._master_port


class BaseModelActor(BaseDistributedActor):
    """Base class for model actors that also carry a DeepSpeed strategy (policy/reference/critic)."""

    def _setup_distributed(self, strategy: DeepspeedStrategy):
        # configure strategy
        self.strategy = strategy
        strategy.setup_distributed()

    def init_model_from_pretrained(self, *args, **kwargs):
        """Load the model; must be implemented by subclasses."""
        raise NotImplementedError()

    def empty_cache(self) -> None:
        """Free and synchronize CUDA memory on this actor."""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def execute_batch(self, method_name: str, all_data, start_idx, end_idx):
        """Process input data by calling specified function for each item in the lists.

        Args:
            method_name (str): Name of the function to execute
            kwargs: Reference to the chunk of data to process

        Returns:
            List[Any]: List of results from function execution
        """

        # Get the first parameter to determine list length
        kwargs = {key: value[start_idx:end_idx] for key, value in all_data.items()}
        first_param = next(iter(kwargs.values()))
        list_length = len(first_param)

        # Verify all parameters have same length
        for param_name, param_value in kwargs.items():
            if len(param_value) != list_length:
                raise ValueError(
                    f"Parameter {param_name} has length {len(param_value)}, expected {list_length}"
                )

        # Get the function to execute
        func = getattr(self, method_name)
        if not callable(func):
            raise ValueError(f"Function {method_name} is not callable")

        results = []
        for i in tqdm(
            range(list_length),
            desc=f"{method_name}",
            disable=not self.strategy.is_rank_0(),
        ):
            # Create kwargs for single item
            sample_kwargs = {
                param_name: param_value[i] for param_name, param_value in kwargs.items()
            }

            result = func(**sample_kwargs)
            results.append(result)

        return results


class RayActorGroup:
    """
    A group of ray actors using deepspeed strategy
    Functions start with 'async' should return list of object refs
    disentangle number of cpus and gpus here

    Args:
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        ray_actor_type (Type[BaseModelActor]): PPO model type that this actor group serve on.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        num_gpus: int,
        ray_actor_type: Type[BaseModelActor],
        pg: PlacementGroup = None,
        num_cpus_per_actor: int = 1,
        num_gpus_per_actor: int = 1,
        duplicate_actors: int = 1,
        resources: Dict[str, float] = None,
        num_resources_per_node: int = None,
    ) -> None:
        self.world_size = num_gpus
        self.ray_actor_type = ray_actor_type
        # duplicate actors is ring_attn_size * tensor_parallel_size
        self.duplicate_actors = duplicate_actors

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node

        self._initiate_actors(pg, num_cpus_per_actor, num_gpus_per_actor)

    def _initiate_actors(
        self,
        pg,
        num_cpus_per_actor: Union[float, int],
        num_gpus_per_actor: Union[float, int],
    ):
        """we need to overwrite the initialization in ray actor group because rule based RM doesn't need gpus"""
        # Use placement group to lock resources for models of same type
        if pg is None:
            if self.world_size > 0:
                bundles = [{"GPU": 1, "CPU": 1} for _ in range(self.world_size)]
            else:
                bundles = [{"GPU": 0, "CPU": 1}]  # rule-based RM only needs cpu

            if self._resources:
                resources_name = list(self._resources.keys())[0]
                for i in range(len(bundles)):
                    bundles[i][resources_name] = self._num_resources_per_node

            pg = placement_group(bundles, strategy="PACK")
            ray.get(pg.ready())
        if pg:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_cpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0
                ),
            ).remote(self.world_size, 0, None, None)
        else:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_cpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
            ).remote(self.world_size, 0, None, None)
        self._actor_handlers = [master_actor]

        # Create worker_actor
        if self.world_size > 1:
            master_addr, master_port = ray.get(
                master_actor.get_master_addr_port.remote()
            )
            for rank in range(1, self.world_size):
                if pg:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_cpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=rank,
                        ),
                    ).remote(self.world_size, rank, master_addr, master_port)
                else:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_cpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                    ).remote(self.world_size, rank, master_addr, master_port)
                self._actor_handlers.append(worker_actor)

    def async_init_model_from_pretrained(
        self,
        *args,
        **kwargs,
    ):
        """Init model from pretrained checkpoint.

        Returns:
            List: list of remote object refs.
        """
        return [
            actor.init_model_from_pretrained.remote(*args, **kwargs)
            for actor in self._actor_handlers
        ]

    def async_save_model(self):
        """Save actor model on rank 0.

        Returns:
            List: list of remote object refs.
        """
        return [actor.save_model.remote() for actor in self._actor_handlers]

    def async_run_method(self, method_name, *args, **kwargs):
        """Call ``method_name`` with the same args/kwargs on every actor in the group.

        Returns:
            List[ray.ObjectRef]: One remote object ref per actor.
        """
        refs = []
        for actor in self._actor_handlers:
            method = getattr(actor, method_name)
            refs.append(method.remote(*args, **kwargs))
        return refs

    def async_run_method_batch(self, method_name, **kwargs):
        """Run method on all actors with batched input data asynchronously using round-robin scheduling.
        Each actor processes one chunk of data at a time. Actors in the same ring / tensor parallel group process the same chunk.

        Args:
            method_name (str): Name of the method to run
            **kwargs: Keyword arguments for the method. Each value should be a list/tensor of the same length.

        Returns:
            List[ray.ObjectRef]: List of remote object references to the results
        """
        # Check if all kwargs parameters are iterable
        for key, value in kwargs.items():
            if not hasattr(value, "__len__"):
                raise ValueError(f"Parameter {key} must be iterable")

        # Get the length of the first parameter as reference
        first_param = next(iter(kwargs.values()))
        total_length = len(first_param)

        # Verify all parameters have the same length
        for key, value in kwargs.items():
            if len(value) != total_length:
                raise ValueError(
                    f"All parameters must have the same length. {key} has length {len(value)}, expected {total_length}"
                )

        # Calculate chunk size based on number of effective actors (considering ring groups)
        num_actors = len(self._actor_handlers)
        effective_actors = num_actors // self.duplicate_actors
        chunk_size = total_length // effective_actors
        assert total_length >= effective_actors, (
            f"Total length {total_length} must be greater than or equal to effective actors {effective_actors}"
        )
        if total_length % effective_actors != 0:
            chunk_size += 1

        all_data_ref = ray.put(kwargs)

        refs = []
        for chunk_idx in range(effective_actors):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_length)

            for j in range(self.duplicate_actors):
                actor_idx = chunk_idx * self.duplicate_actors + j
                actor = self._actor_handlers[actor_idx]

                refs.append(
                    actor.execute_batch.remote(
                        method_name, all_data_ref, start_idx, end_idx
                    )
                )

        return refs
