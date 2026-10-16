from turtle import up
import torch
from torch import nn
from torch.nn import functional as F


def _make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    h = 2 * torch.Tensor(list(range(0, shape[-2], 1))) / (shape[-2] - 1.0) - 1.0
    w = 2 * torch.Tensor(list(range(0, shape[-1], 1))) / (shape[-1] - 1.0) - 1.0
    ret = torch.stack(torch.meshgrid(h, w), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class LIIFDecoder(nn.Module):
    def __init__(self, in_channels=256, num_class=19, up_ratio=4, hidden_list=[32, 32, 32, 32]):
        super().__init__()
        self.imnet = MLP(in_dim=in_channels+6, out_dim=num_class, hidden_list=hidden_list)
        self.up_ratio = up_ratio
        self.dropout=nn.Dropout2d(0.1)
        # self.dropout=nn.Dropout(0.1)

    def forward(self, feat, feat_coord=None, target_size=None):
        """
        coord in [h, w, 2] in hw
        """
        feat = self.dropout(feat)
        if target_size == None:
            if feat.size(-2) % 2 == 1:
                target_size = [i * self.up_ratio - self.up_ratio + 1 for i in feat.shape[-2:]]
            else:
                target_size = [i * self.up_ratio for i in feat.shape[-2:]]
                
        vh_lst = [-1, 1]
        vw_lst = [-1, 1]
        eps_shift = 1e-6

        rh = 2 / feat.shape[-2] / 2 
        rw = 2 / feat.shape[-1] / 2 
        if feat_coord == None:
            feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:]) # 
        coord = make_coord(target_size).to(feat.device)
        coord = coord.repeat(feat.size(0), 1, 1)
        # print(coord.shape)
        # cell = torch.ones_like(hr_coord)
        # cell[:, 0] *= 2 / seg.shape[-2] 
        # cell[:, 1] *= 2 / seg.shape[-1]
        cell = torch.ones_like(torch.Tensor(coord.shape)).to(feat.device)
        cell[:, :, 0] *= 2 / target_size[-2] 
        cell[:, :, 1] *= 2 / target_size[-1]

        preds = []
        areas = []
        for vh in vh_lst:
            for vw in vw_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vh * rh + eps_shift
                coord_[:, :, 1] += vw * rw + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord, coord], dim=-1)

                # if self.cell_decode:
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        # print(tot_area.max(), tot_area.min(), tot_area.mean())
        # print(tot_area)
        # print(tot_area.shape)
        # if self.local_ensemble:
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0

        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        
        # pred_224 = torch.sigmoid(ret) # [6, 1, 224, 224]

        # images = {}
        # images['out_224'] = ret
        # images['pred_224'] = pred_224
        
        # return images        
        return ret.view(feat.size(0), target_size[-2], target_size[-1], -1).permute(0, 3, 1, 2)

def test_LIIF_decoder():
    feat = torch.rand(4, 256, 3 ,3)
    m = LIIFDecoder(hidden_list=[])
    y = m(feat)
    print(y.shape)
    
def test_make_coord():
    feat = torch.rand(1, 1, 3 ,3)
    coord = make_coord(feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
    # coord = make_coord(feat.shape[-2:], flatten=False).permute(2, 0, 1)
    print(coord)
    print(coord.shape)

if __name__ == '__main__':
    # test_LIIF_decoder()
    test_make_coord()
   