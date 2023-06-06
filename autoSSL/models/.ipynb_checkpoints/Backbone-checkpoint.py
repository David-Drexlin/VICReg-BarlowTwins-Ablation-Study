
def pipe_backbone(backbone="resnet18"):
    if backbone == "resnet18":
        return nn.Sequential(*list(models.resnet18().children())[:-1]),512
    elif backbone == "resnet18_pretrained":
        return nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).children())[:-1]),512
    elif backbone == "resnet50":
        return nn.Sequential(*list(models.resnet50().children())[:-1]),2048
    elif backbone == "resnet50_pretrained":
        return nn.Sequential(*list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-1]),2048
    elif backbone == "efficientnet_b5":
        return nn.Sequential(*list(models.efficientnet_b5().children())[:-1]),2048
    elif backbone == "efficientnet_b5_pretrained":
        return nn.Sequential(*list(models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1).children())[:-1]),2048
    elif backbone == "mobilenet_v3":
        return nn.Sequential(*list(models.mobilenet_v3_large().children())[:-1]),960
    elif backbone == "mobilenet_v3_pretrained":
        return nn.Sequential(*list(models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1).children())[:-1]),960
    elif backbone == "vit_64":
        return torchvision.models.VisionTransformer(
                image_size=64,
                patch_size=16,
                num_layers=12,
                num_heads=6,
                hidden_dim=384,
                mlp_dim=384 * 4
            ),1000
    elif backbone == "vit_224":
        return torchvision.models.VisionTransformer(
                image_size=224,
                patch_size=16,
                num_layers=12,
                num_heads=6,
                hidden_dim=384,
                mlp_dim=384 * 4
            ),1000
    else:
        raise ValueError("Invalid backbone")
        