from argparse import ArgumentParser
from pnpxai.visualizer.backend.app import create_app
from pnpxai.visualizer import config


def main(args: dict):
    app = create_app(
        address=args.get('address', None),
        port=args.get('port', None),
        password=args.get('password', None)
    )
    app.run()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--address', type=str, default=config.MGR_DFT_ADDRESS)
    parser.add_argument('--port', type=int, default=config.MGR_DFT_ADDRESS)
    parser.add_argument('--password', type=str, default=config.MGR_DFT_ADDRESS)

    main(vars(parser.parse_args()))
