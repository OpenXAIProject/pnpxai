from argparse import ArgumentParser
from pnpxai.visualizer import config
from pnpxai.visualizer.proc_manager.server import Server


def main(args: dict):
    server = Server(
        address=args.get('address', None),
        port=args.get('port', None)
    )
    server.serve_forever()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--address', type=str, default=config.MGR_DFT_ADDRESS)
    parser.add_argument('--port', type=int, default=config.MGR_DFT_ADDRESS)

    main(vars(parser.parse_args()))
