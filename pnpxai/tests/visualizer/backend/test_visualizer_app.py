import pytest
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset

from pnpxai import Project
from pnpxai.visualizer.backend.app import create_app
from pnpxai.visualizer.server import Server
from pnpxai.utils import class_to_string


class DummyDataset(Dataset):
    def __init__(self, data):
        super(DummyDataset, self).__init__()
        self._data = data

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class TestVisualizerApp:
    @pytest.fixture
    def app_projects(self):
        model = nn.Linear(1, 1)
        loader = DataLoader(DummyDataset([
            (Tensor([0, 0, 0]), Tensor([0]))
        ]))

        project1 = Project('project1')
        project1.create_auto_experiment(
            model=model,
            data=loader,
            name="experiment1"
        )

        return Server().get_projects_map()

    @pytest.fixture
    def app(self, app_projects):
        app = create_app(app_projects)
        yield app
        Server().reset()

    @pytest.fixture
    def client(self, app):
        return app.test_client()

    def test_api_get_projects(self, client, app_projects):
        response = client.get(f'/api/projects/')
        payload = response.get_json()
        assert payload['message'] == 'Success'
        data = payload['data']
        for idx, (name, project) in enumerate(app_projects.items()):
            data_project = data[idx]
            assert data_project['id'] == name
            for exp_idx, (exp_name, exp) in enumerate(project.experiments.items()):
                data_experiment = data_project['experiments'][exp_idx]
                assert data_experiment['name'] == exp_name
                explainers = [class_to_string(explainer)
                              for explainer in exp.all_explainers]
                assert data_experiment['explainers'] == explainers

    def test_api_get_experiments(self, client, app_projects):
        project = list(app_projects.values())[0]
        exp_name, experiment = list(project.experiments.items())[0]
        response = client.get(
            f'/api/projects/{project.name}/experiments/{exp_name}/')
        print(response)
        payload = response.get_json()
        assert payload['message'] == 'Success'

    def test_api_get_experiments(self, client, app_projects):
        project = list(app_projects.values())[0]
        exp_name, experiment = list(project.experiments.items())[0]
        response = client.get(
            f'/api/projects/{project.name}/experiments/{exp_name}/')
        print(response)
        payload = response.get_json()
        assert payload['message'] == 'Success'

    def test_api_get_experiment_inputs(self, client, app_projects):
        project = list(app_projects.values())[0]
        exp_name, experiment = list(project.experiments.items())[0]
        response = client.get(
            f'/api/projects/{project.name}/experiments/{exp_name}/inputs/')
        payload = response.get_json()
        assert payload['message'] == 'Success'

    def test_api_get_project_models(self, client, app_projects):
        project = list(app_projects.values())[0]
        exp_id = 0
        exp_name, experiment = list(project.experiments.items())[exp_id]
        response = client.get(
            f'/api/projects/{project.name}/models/')
        payload = response.get_json()
        assert payload['message'] == 'Success'
        assert payload['data'][exp_id]['name'] == class_to_string(
            experiment.model)
