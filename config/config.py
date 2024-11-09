import yaml
import os 
yaml_path = 'config/config.yaml'
with open(yaml_path, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    if ("__include__" in cfg):
        for include_file in cfg["__include__"]:
            with open(include_file, 'r') as f:
                include_config = yaml.load(f, Loader=yaml.FullLoader)
                try:
                    cfg.update(include_config)
                except:
                    print("Error: failed to include {}".format(include_file))   