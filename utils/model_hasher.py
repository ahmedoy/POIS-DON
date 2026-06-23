import torch
import hashlib
import tempfile
import os


def hash_model_on_disk(model: torch.nn.Module, chunk_size: int = 65536):
    hash_gen = hashlib.sha256()

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = os.path.join(tmpdir, "model_state.pt")

        # FIX: Move to CPU and use a stable serialization format
        # This ensures the hash is the same regardless of GPU vs CPU
        torch.save(model.state_dict(), temp_path, _use_new_zipfile_serialization=True)

        with open(temp_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hash_gen.update(chunk)

    return hash_gen.hexdigest()


def get_model_id(model: torch.nn.Module, num_characters=15):
    full_hash = hash_model_on_disk(model)
    return full_hash[:num_characters]  # Returns first num_characters characters
