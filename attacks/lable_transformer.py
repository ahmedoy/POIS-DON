class LableTransformer:
    def __init__(self, lable_mapping: dict[int, int]):
        self.lable_mapping = lable_mapping

    def __call__(self, x: int) -> int:
        return self.lable_mapping[x]
