from rfdetr import RFDETRSegMedium

model = RFDETRSegMedium(
    pretrain_weights="/home/sakif/cattle_logs/checkpoint_best_ema_B.pth",
    num_classes=1,
)
print("Loaded successfully")
