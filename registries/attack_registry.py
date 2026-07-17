from attacks.base_dataloader_attack import DataloaderBasedAttackBase
from attacks.clean_basic.clean_basic import CleanBasic
from attacks.sig_attack.sig import SIG
from attacks.badnet.badnet import BadNet
from attacks.wanet_no_noise.wanet_no_noise import WaNetNoNoise

ATTACK_REGISTRY: dict[str, DataloaderBasedAttackBase] = {
    "clean_basic": CleanBasic(),
    "sig": SIG(),
    "badnet": BadNet(),
    "wanet_no_noise": WaNetNoNoise(),
}
