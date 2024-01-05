from pnpxai.visualizer.server import Server
from pnpxai import Project


class TestServer:
    def test_singleton_init(self):
        server1 = Server()
        server2 = Server()
        assert id(server1) == id(server2)

    def test_server_register_method(self):
        server = Server()
        project1 = Project('project1')
        assert server._projects[-1].name == project1.name
        project2 = Project('project2')
        assert server._projects[-1].name == project2.name

        assert len(server._projects) == 2
        server.reset()
        assert len(server._projects) == 0


    def test_server_unregister_method(self):
        server = Server()
        project1 = Project('project1')
        assert len(server._projects) == 1

        server.unregister(project1)
        assert len(server._projects) == 0

        server.reset()

    def test_server_unregister_by_name_method(self):
        server = Server()
        project1 = Project('project1')
        assert len(server._projects) == 1

        server.unregister_by_name(project1.name)
        assert len(server._projects) == 0

        server.reset()

    def test_server_get_projects_map_method(self):
        server = Server()
        project1 = Project('project1')

        assert server.get_projects_map() == {
            project1.name: project1
        }

        server.reset()