from multiprocessing import Process, Manager
from pnpxai import Project
from tutorials.helpers import get_imagenet_dataset, get_torchvision_model
import torch
from torch.utils.data import DataLoader


def f(d, l):
    print("Inside of the process, ", d)
    print(d)
    project = d['project']
    project.experiments[0].run()
    d['project'] = project


if __name__ == '__main__':
    manager = Manager()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, transform = get_torchvision_model("resnet18")

    dataset = get_imagenet_dataset(transform, subset_size=100)
    loader = DataLoader(dataset, batch_size=8)

    project = Project('test_project')
    experiment = project.create_auto_experiment(
        model,
        loader,
        name='test_experiment'
    )

    d = manager.dict()
    l = manager.list()
    d['project'] = project

    p = Process(target=f, args=(d, l))
    p.start()
    p.join()

    print(d)

# from multiprocessing import Process, Manager

# class Test:
#     def __init__(self, a):
#         self.a = a

# def f(d, l):
#     print("In SUB, BEFORE:", d["test"].a)
#     test = d["test"]
#     test.a.append(23)
#     d["test"] = test
#     print("In SUB, AFTER:", d["test"].a)

# if __name__ == '__main__':
#     manager = Manager()

#     d = manager.dict()
#     d['test'] = Test([1, 2])
#     l = manager.list(range(10))

#     p = Process(target=f, args=(d, l))
#     p.start()
#     p.join()

#     print(d['test'].a)
#     print(l)