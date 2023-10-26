# from abc import abstractmethod
import os
import json
from dataclasses import dataclass, asdict, astuple
from itertools import product
from functools import partial
from typing import Literal, Optional
from tqdm import tqdm

import torch
import pandas as pd
from torch.utils.data import DataLoader

from .database import Database, _default_root_dir, mkdir
from .manager import ObjectManager
from ..core.explainers.lrp import SUPPORTED_LRP
from ..core.explainers.gradcam import SUPPORTED_GRADCAM


@dataclass(repr=False)
class _PnpxaiObject:
    id: Optional[int] = None
    name: Optional[str] = None

    to_dict = asdict
    to_tuple = astuple

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"
    
    @classmethod
    @property
    def objects(cls):
        return ObjectManager(cls=cls)

    @classmethod
    @property
    def _table_name(cls):
        return f"{cls.__name__.lower()}s"

    @classmethod
    @property
    def _relation_id_key(cls):
        return f"{cls.__name__.lower()}_id"

    @classmethod
    def _relation_table_name(cls, target):
        return '_'.join([cls._relation_id_key, 'to', target._relation_id_key])

    def save(self):
        obj = self.objects.update(**self.to_dict()) if self.id != None else self.objects.create(**self.to_dict())
        self = obj
        return self


@dataclass(repr=False)
class Dataset(_PnpxaiObject):
    uri: Optional[str] = None
    origin: Literal["torch", "pandas"] = "torch"

    def load(self, **kwargs):
        if self.origin == "torch":
            return torch.load(self.uri, **kwargs)

    def load_random_samples(self, n_samples):
        dataset = self.load()
        dataloader = DataLoader(dataset, batch_size=n_samples, shuffle=True)
        return next(iter(dataloader))


@dataclass(repr=False)
class Model(_PnpxaiObject):
    uri: Optional[str] = None
    origin: Literal["torch", "tensorflow"] = "torch"

    def load(self, **kwargs):
        if self.origin == "torch":
            return torch.load(self.uri, **kwargs)

    def is_lrp_applicable(self):
        return True

    def is_cam_applicable(self):
        return self._contains_submodule(startswith="Conv", module=self.load())

    def get_forwarding_order(self, input_shape):
		# https://github.com/frgfm/torch-cam/blob/main/torchcam/methods/_utils.py#L15
        model = self.load()
        model.eval()
        forwarding_order = list()
        def _record_output_shape(
            module: "nn.Module",
            input: "torch.Tensor",
            output: "torch.Tensor",
            name: Optional[str] = None
        ) -> None:
            forwarding_order.append((name, output.shape))
        
        hook_handles = list()
		
        for n, m in model.named_modules():
            hook_handles.append(
                m.register_forward_hook(
                    partial(_record_output_shape, name=n)
            ))
		
        _ = model(torch.zeros(
			(1, *input_shape),
			device = next(model.parameters()).data.device
		))
        
        for handle in hook_handles:
            handle.remove()
        return forwarding_order

    def _contains_submodule(self, startswith: str, module: Optional["nn.Module"] = None):
        children = list(module.children())
        if children:
            for child in children:
                if self._contains_submodule(startswith=startswith, module=child):
                    return True
            return False
        else:
            return module.__class__.__name__.startswith(startswith)
            
    def find_target_layer_for_cam(self, input_shape):
		# https://github.com/frgfm/torch-cam/blob/main/torchcam/methods/_utils.py#L15
        model = self.load()
        if not self._contains_submodule(startswith='Conv', module=model):
            return
        inspected = list()
        forwarding_order = self.get_forwarding_order(input_shape)
        _is_changing_dimension = lambda output_shape: (
            len(output_shape) != (len(input_shape)+1)
        )
        _is_final_pooling = lambda output_shape: (
            any(v == 1 for v in output_shape[2:])
        )
        _is_inspected = lambda layer_name: (
            any(layer_name.startswith(f"{i}.") for i in inspected)
        )
        # [GH] I have no idea why the following lambda function not working on L158
        # _is_convolutional = lambda layer_name: (
        #     self._contains_submodule(
        #         startswith = 'Conv',
        #         module = eval(f"model.{layer_name}"),
        # ))
        found = False
        while forwarding_order:
            layer_name, output_shape = forwarding_order.pop()
            if layer_name == "":
                continue
            if all([
                not _is_inspected(layer_name),
                not _is_final_pooling(output_shape),
                not _is_changing_dimension(output_shape),
                self._contains_submodule(startswith="Conv", module=eval(f"model.{layer_name}")),
                # _is_convolutional(layer_name)
            ]):
                found = True
                break
            inspected.append(layer_name)
        if found:
            return layer_name
        return


@dataclass(repr=False)
class Project(_PnpxaiObject):
    # fields for task obj
    
    @property
    def datasets(self):
        return ObjectManager(Dataset, parent=self)

    @property
    def models(self):
        return ObjectManager(Model, parent=self)

    @property
    def tasks(self):
        return ObjectManager(Task, parent=self)

    def add_dataset(self, dataset):
        assert isinstance(dataset, Dataset)
        self._relate(dataset)
        return self

    def add_model(self, model):
        assert isinstance(model, Model)
        self._relate(model)
        return self

    def add_task(self, task):
        assert isinstance(task, Task)
        self._relate(task)
        return self

    def _relate(self, target):
        with Database(table_name=self._relation_table_name(target)) as table:
            record = {
                self._relation_id_key: self.id,
                target._relation_id_key: target.id
            }
            selected = table.select(**record)
            if len(selected) == 0:
                table.insert(record)

    # [GH] method name is somewhat awkward and not intuitive
    def list_all_task_data(self):
        return list(product(
            self.datasets.all(),
            self.models.all(),
        ))

    def update_tasks(self, indices=None, names=None, store_root=None):
        task_data = self.list_all_task_data()
        if indices:
            task_data = [task_data[i] for i in indices]
        for i, (dataset, model) in enumerate(task_data):
            name = names[i] if names else "_".join([
                "task",
                str(dataset.id).zfill(2),
                str(model.id).zfill(2),
            ])
            task, created = Task.objects.get_or_create(
                name = name,
                project_id = self.id,
                dataset_id = dataset.id,
                model_id = model.id,
            )
            self.add_task(task)
        self._sync_store(store_root)

    def _sync_store(self, root_directory=None):
        root_dir = root_directory or _default_root_dir()
        proj_dir = os.path.join(root_dir, self.name)
        if not os.path.exists(proj_dir):
            mkdir(root_dir, self.name)
        for task in self.tasks.all():
            task_dir = os.path.join(proj_dir, task.name)
            if not os.path.exists(task_dir):
                mkdir(proj_dir, task.name)
            with open(os.path.join(task_dir, f"{task.name}.json"), "w") as f:
                json.dump(task.to_dict(), f)


@dataclass(repr=False)
class Task(_PnpxaiObject):
    project_id: Optional[int] = None
    dataset_id: Optional[int] = None
    model_id: Optional[int] = None

    @property
    def project(self):
        return Project.objects.get(id=self.project_id)

    @property
    def dataset(self):
        return Dataset.objects.get(id=self.dataset_id)

    @property
    def model(self):
        return Model.objects.get(id=self.model_id)

    @property
    def input_shape(self):
        x, y = self.dataset.load_random_samples(n_samples=1)
        return tuple(x[0].shape)
        
    def _next_input_id(self):
        indices = [
            int(dir_name[5:]) for dir_name in os.listdir(self.get_logging_dir())
            if dir_name.startswith("input")
        ]
        if not indices:
            return 0
        return max(indices) + 1
        
    def get_logging_dir(self, root_directory=None):
        root_dir = root_directory or _default_root_dir()
        return os.path.join(root_dir, self.project.name, self.name)

	# [TO-DO]
    def applicable_explainers(self, input_shape=None):
        input_shape = input_shape or self.input_shape
        applicables = list()
        model = self.model.load()
        if self.model.is_lrp_applicable:
            applicables += [cls(model) for cls in SUPPORTED_LRP.values()]
        if self.model.is_cam_applicable:
            target_layer = self.model.find_target_layer_for_cam(input_shape)
            applicables += [cls(model, target_layer=target_layer) for cls in SUPPORTED_GRADCAM.values()
            ]
        return applicables

    def run_all_applicables(self, inputs, targets, log=True, **hyperparameters):
        input_shape = inputs[0].shape
        applicables = self.applicable_explainers(input_shape)

        # get explanations
        attributions = list()
        pbar = tqdm(applicables, total=len(applicables))
        for explainer in pbar:
            pbar.set_description(f"Running {explainer.__class__.__name__}")
            attributions.append(explainer.attribute(inputs, targets))
        if not log:
            return attributions

        # log
        from torchvision.transforms.functional import to_pil_image
        from torchvision.transforms import Normalize
        transform = self.dataset.load().transform
        mean = torch.Tensor(transform.mean)
        std = torch.Tensor(transform.std)
        denormalize = Normalize((-1*mean/std), (1./std))
        for i, (input, target) in enumerate(zip(inputs, targets)):
            # log original input
            logging_dir = self.get_logging_dir()
            next_id = self._next_input_id()
            mkdir(logging_dir, f"input{next_id}")
            input_dir = os.path.join(logging_dir, f"input{next_id}")
            
            denormed = denormalize(input)
            save_path = os.path.join(input_dir, "original_input.png")
            to_pil_image(denormed).save(save_path)

            # log attributions
            attrs_i = [
                attr[i] if len(attr[i].shape) == 2 else attr[i].mean(0)
                for attr in attributions
            ]
            for explainer, attr in zip(applicables, attrs_i):
                attr = (attr - attr.min()) / (attr.max() - attr.min())
                save_path = os.path.join(input_dir, f"{explainer.__class__.__name__}.png")
                try:
                    to_pil_image(attr).save(save_path)
                except:
                    return attr
        return attributions