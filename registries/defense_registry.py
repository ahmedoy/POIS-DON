from defenses.base_defense import DefenseBase
from defenses.poisdon.poisdon import Poisdon

DEFENSE_REGISTRY: dict[str, DefenseBase] = {
    "poisdon": Poisdon(),
}
