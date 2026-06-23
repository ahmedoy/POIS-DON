from attacks.base_dataloader_attack import DataloaderBasedAttackBase
from attacks.clean_basic.clean_basic import CleanBasic

ATTACK_REGISTRY: dict[str, DataloaderBasedAttackBase] = {"clean_basic": CleanBasic()}
