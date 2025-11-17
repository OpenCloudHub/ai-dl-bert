from logging import getLogger
from typing import List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger

log = getLogger()


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    # Check if callback configurations are provided
    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    # Validate the type of callbacks configuration
    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig object!")

    # Iterate over each callback configuration and instantiate it
    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    loggers: List[Logger] = []

    # Check if logger configurations are provided
    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return loggers

    # Validate the type of logger configuration
    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    # Iterate over each logger configuration and instantiate it
    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))
        else:
            log.warning(f"Unable to instantiate log config <{lg_conf}>! Skipping...")

    return loggers
