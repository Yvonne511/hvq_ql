import hydra
import wandb
import submitit_patch
from omegaconf import OmegaConf

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg):
    wandb.init(
        project="hydra-greene-demo",
        config=OmegaConf.to_container(cfg),
    )
    result = cfg.x ** 2 + cfg.y ** 2
    wandb.log({"result": result})
    print(cfg)
    print("Result:", result)
    wandb.finish()
    return result

if __name__ == "__main__":
    main()
