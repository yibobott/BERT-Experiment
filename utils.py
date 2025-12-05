import argparse
import yaml


def load_yaml_config():
    """
    Parse command line to get ``--config`` path and load a YAML config file

    Returns:
        dict: Configuration dictionary parsed from the YAML file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg
