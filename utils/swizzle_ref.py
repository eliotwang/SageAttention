# file: test_swizzle_roundtrip.py
import torch

# 定义 permute 索引（来自 kernel）
P16     = torch.tensor([0,1,4,5,8,9,12,13, 2,3,6,7,10,11,14,15], dtype=torch.long)
P16_INV = torch.tensor([0,1,8,9,2,3,10,11, 4,5,12,13,6,7,14,15], dtype=torch.long)

def swizzle_seq16_bhnd_to_bhdn(v_bhnd: torch.Tensor):
    """
    (b,h,n,d) -> pad n 到 64 的倍数 -> (b,h,n_pad,d)
    -> 每 16 做 P16 permute -> (b,h,d,n_pad)
    """
    b,h,n,d = v_bhnd.shape
    n_pad = (n + 63)//64*64
    x = torch.zeros((b,h,n_pad,d), dtype=v_bhnd.dtype, device=v_bhnd.device)
    x[...,:n,:] = v_bhnd
    x = x.view(b,h,n_pad//16,16,d).index_select(3, P16.to(x.device)).view(b,h,n_pad,d)
    return x.permute(0,1,3,2).contiguous()  # (b,h,d,n_pad)

def unswizzle_seq16_bhdn_to_bhnd(v_bhdn: torch.Tensor, n: int):
    """
    (b,h,d,n_pad) -> (b,h,n_pad,d)
    -> 每 16 做 P16_INV -> (b,h,n_pad,d)
    -> 裁剪到原始 n -> (b,h,n,d)
    """
    b,h,d,n_pad = v_bhdn.shape
    assert n_pad % 16 == 0, "n_pad 必须能被16整除"
    x = v_bhdn.permute(0,1,3,2).contiguous()  # (b,h,n_pad,d)
    x = x.view(b,h,n_pad//16,16,d).index_select(3, P16_INV.to(x.device)).view(b,h,n_pad,d)
    x = x[...,:n,:]
    return x.contiguous()

@torch.no_grad()
def round_trip_check(v_ref: torch.Tensor):
    """验证 v -> swizzle -> unswizzle 是否能还原"""
    v_fp8layout = swizzle_seq16_bhnd_to_bhdn(v_ref)
    v_back = unswizzle_seq16_bhdn_to_bhnd(v_fp8layout, n=v_ref.size(2))
    diff = (v_back - v_ref).abs()
    print("round-trip MAE:", diff.mean().item(), "max:", diff.max().item())
    # 逐列余弦（沿 n）
    A = v_ref[0,0,:,:].to(torch.float32)  # (n,d)
    B = v_back[0,0,:,:].to(torch.float32)
    num = (A*B).sum(0)
    den = (A.square().sum(0).sqrt() * B.square().sum(0).sqrt()).clamp_min(1e-12)
    cos = (num/den).cpu()
    print("per-col cosine mean:", float(cos.mean()), 
          "min:", float(cos.min()), "max:", float(cos.max()))

if __name__ == "__main__":
    torch.manual_seed(0)
    b,h,n,d = 1, 30, 8866, 64   # n 不是 16 的倍数，测试 pad/反pad
    v = torch.randn(b,h,n,d, dtype=torch.float16, device="cuda")
    print("input shape:", v.shape)
    round_trip_check(v)
