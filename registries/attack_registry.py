from attacks.base_dataloader_attack import DataloaderBasedAttackBase
from attacks.clean_basic.clean_basic import CleanBasic
from attacks.sig_attack.sig import SIG

ATTACK_REGISTRY: dict[str, DataloaderBasedAttackBase] = {
    "clean_basic": CleanBasic(),
    "sig": SIG(),
}
