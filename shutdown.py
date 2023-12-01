from pnpxai.visualizer.proc_manager.client import Client

client  = Client()
client.connect_to_server()
print(client.get_all_projects())