import sys
import yaml
import torch


def load_yaml_config(path):
    with open(path, 'r') as f:
        config = yaml.full_load(f)
    return config


def save_config_to_yaml(config, path):
    assert path.endswith(".yaml")
    with open(path, "w") as f:
        f.write(yaml.dump(config))


def write_args(args, path):
    args_dict = dict((name, getattr(args, name))
                     for name in dir(args) if not name.startswith("_"))
    with open(path, "a") as args_file:
        args_file.write(f"==> torch version: {torch.__version__}\n")
        args_file.write(
            f"==> cudnn version: {torch.backends.cudnn.version()}\n")
        args_file.write("==> Cmd:\n")
        args_file.write(str(sys.argv))
        args_file.write("\n==> args:\n")
        for k, v in sorted(args_dict.items()):
            args_file.write(f"  {str(k)}: {str(v)}\n")
        args_file.close()
