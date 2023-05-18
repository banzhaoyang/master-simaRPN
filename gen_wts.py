import torch
import struct
pt_file = "siamese_rpn_r50_20e_lasot_20220420_181845-dd0f151e.pth"

model = torch.load(pt_file, map_location="cpu")  # load to FP32

# simarpn
with open('simaRPN++.wts', 'w') as f:
    f.write('{}\n'.format(len(model["state_dict"].keys())))
    for k, v in model["state_dict"].items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f', float(vv)).hex())
        f.write('\n')