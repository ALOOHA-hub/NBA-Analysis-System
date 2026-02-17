from utils.config_loader import cfg
from core.pipline import Pipeline

if __name__ == "__main__":
    pipeline = Pipeline(cfg['settings'])
    pipeline.run()
