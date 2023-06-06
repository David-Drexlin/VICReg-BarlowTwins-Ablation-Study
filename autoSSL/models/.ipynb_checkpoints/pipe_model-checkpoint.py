from autoSSL.models import BarlowTwins, BYOL, MoCo, SimCLR, SimSiam, VICReg 
# Function to get the model
def pipe_model(name="InputYourModelName", config=None, **kwargs):
    if config is not None:
        backbone = config["backbone"]
        stop_gradient = config["stop_gradient"]
        prjhead_dim = config["prjhead_dim"]
        name=config["model"]
        if name == "MoCo":
            return MoCo(backbone=backbone, stop_gradient=stop_gradient, prjhead_dim=prjhead_dim)
        elif name == "BYOL":
            return BYOL(backbone=backbone, stop_gradient=stop_gradient, prjhead_dim=prjhead_dim)
        elif name == "SimCLR":
            return SimCLR(backbone=backbone, stop_gradient=stop_gradient, prjhead_dim=prjhead_dim)
        elif name == "SimSiam":
            return SimSiam(backbone=backbone, stop_gradient=stop_gradient, prjhead_dim=prjhead_dim)
        elif name == "BarlowTwins":
            return BarlowTwins(backbone=backbone, stop_gradient=stop_gradient, prjhead_dim=prjhead_dim)
        elif name == "VICReg":
            return VICReg(backbone=backbone, stop_gradient=stop_gradient, prjhead_dim=prjhead_dim)
        else:
            raise ValueError(f"Unknown model name: {name}")

    else:
        # Use the original implementation if config is not provided
        if name == "MoCo":
            return MoCo(**kwargs)
        elif name == "BYOL":
            return BYOL(**kwargs)
        elif name == "SimCLR":
            return SimCLR(**kwargs)
        elif name == "SimSiam":
            return SimSiam(**kwargs)
        elif name == "BarlowTwins":
            return BarlowTwins(**kwargs)
        elif name == "VICReg":
            return VICReg(**kwargs)
        else:
            raise ValueError(f"Unknown model name: {name}")
