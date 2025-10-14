import torch

if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("name:", p.name)
    print("major,minor:", p.major, p.minor)
else:
    print("No cuda available to torch")
