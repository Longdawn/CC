"""Evaluation entrypoint script."""
import argparse
from utils.io import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    print('Evaluate using config:', args.config)

if __name__ == '__main__':
    main()
