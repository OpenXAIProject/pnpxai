from pnpxai.web.server.server import Server
from pnpxai.web.server.backend.app import create_app

def main():
    server = Server()
    print("Server runs")
    server.serve_forever()
    print("Backend runs")
    app = create_app(server)
    app.run()

if __name__ == '__main__':
    main()