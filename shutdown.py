from pnpxai.visualizer.server.client import Client

client  = Client()
client.connect_to_server()
print(client.get_all_projects())