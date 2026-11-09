import torch
from cleanfid import fid


def clean_inception_extractor(path, device):
    # File-object loading avoids C++ fopen limitations on Unicode Windows paths.
    with path.open("rb") as stream:
        base = torch.jit.load(stream).eval().to(device)

    def extract(images):
        if images.shape[-2:] != (299, 299):
            raise ValueError("Clean-FID preprocessing must resize inputs to 299×299")
        with torch.inference_mode():
            return base.layers.forward((images - 128) / 128).view(images.shape[0], 2048)

    return extract


def clean_features(folder, extractor, device, batch_size=4):
    # clean-fid's upper/lowercase glob patterns duplicate images on Windows.
    files = sorted(str(path) for path in folder.rglob("*")
                   if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg"))
    return fid.get_files_features(files, extractor, num_workers=0, batch_size=batch_size,
                                  device=torch.device(device), mode="clean", fdir=str(folder))
