import pytest
from torch import nn
from torch.utils.data import DataLoader

from pnpxai import Project
from pnpxai.visualizer.backend.app import create_app
from pnpxai.visualizer.server import Server
from pnpxai.utils import class_to_string


class TestVisualizerApp:
    @pytest.fixture
    def app_projects(self):
        model = nn.Linear(1, 1)
        loader = DataLoader([])

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
                explainers = [class_to_string(explainer) for explainer in exp.all_explainers]
                assert data_experiment['explainers'] == explainers