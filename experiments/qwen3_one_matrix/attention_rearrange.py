"""Opt-in experimental SDPA adapter. Never imported by normal listener startup."""
from contextlib import contextmanager

class RepeatedEfficientAttention:
    def __init__(self,torch):
        self.torch=torch;self.calls=0;self.signatures={}

    def __call__(self,module,query,key,value,attention_mask,dropout=0.0,scaling=None,is_causal=None,**kwargs):
        torch=self.torch
        if kwargs.get('output_attentions') or kwargs.get('head_mask') is not None:
            raise ValueError('Experimental adapter does not support attention weights/head masks')
        qheads=query.shape[1];kheads=key.shape[1]
        if qheads%kheads or value.shape[1]!=kheads:raise ValueError('Incompatible KV head counts')
        if qheads!=kheads:
            key=key.repeat_interleave(qheads//kheads,dim=1)
            value=value.repeat_interleave(qheads//kheads,dim=1)
        if attention_mask is not None and attention_mask.ndim==4:
            attention_mask=attention_mask[:,:,:,:key.shape[-2]]
        if is_causal is None:
            is_causal=query.shape[2]>1 and attention_mask is None and getattr(module,'is_causal',True)
        from torch.nn.attention import sdpa_kernel,SDPBackend
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            out=torch.nn.functional.scaled_dot_product_attention(query,key,value,attn_mask=attention_mask,
                dropout_p=dropout,scale=scaling,is_causal=is_causal,enable_gqa=False)
        self.calls+=1
        label=('prefill' if query.shape[2]>1 else 'decode')+('-masked' if attention_mask is not None else '-unmasked')
        if label not in self.signatures:self.signatures[label]=dict(query_shape=list(query.shape),query_stride=list(query.stride()),
            repeated_key_shape=list(key.shape),dtype=str(query.dtype),mask_shape=None if attention_mask is None else list(attention_mask.shape),
            is_causal=is_causal,scale=scaling,dropout=dropout)
        return out.transpose(1,2).contiguous(),None

    def receipt(self):
        return dict(variant='repeat_kv_efficient_v1',forced_backend='EFFICIENT_ATTENTION',fallback=False,
                    calls=self.calls,signatures=self.signatures)

@contextmanager
def install(torch):
    # Keep config='sdpa' so Transformers' causal-mask preparation stays unchanged.
    # The benchmark worker is isolated; restore even that process's entry on exit.
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original=ALL_ATTENTION_FUNCTIONS['sdpa'];adapter=RepeatedEfficientAttention(torch)
    ALL_ATTENTION_FUNCTIONS['sdpa']=adapter
    try:yield adapter
    finally:ALL_ATTENTION_FUNCTIONS['sdpa']=original

def validate(torch):
    from types import SimpleNamespace
    from torch.nn.attention import sdpa_kernel,SDPBackend
    torch.manual_seed(8217);rows=[]
    adapter=RepeatedEfficientAttention(torch)
    for label,nq,nk,masked in [('prefill',256,256,False),('decode',1,257,False),('masked',7,263,True)]:
        q=torch.randn(1,nq,32,128,device='cuda',dtype=torch.float16).transpose(1,2)
        k=torch.randn(1,8,nk,128,device='cuda',dtype=torch.float16)
        v=torch.randn_like(k)
        mask=torch.ones(1,1,nq,nk+3,dtype=torch.bool,device='cuda') if masked else None
        if mask is not None:mask[:,:,:,:5]=False
        causal=nq>1 and mask is None
        with sdpa_kernel([SDPBackend.MATH]):
            expected=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=None if mask is None else mask[:,:,:,:nk],
                is_causal=causal,scale=128**-.5,enable_gqa=True).transpose(1,2).contiguous()
        actual,_=adapter(SimpleNamespace(is_causal=True),q,k,v,mask,scaling=128**-.5)
        error=(actual-expected).abs().max().item();finite=bool(torch.isfinite(actual).all())
        rows.append(dict(case=label,max_abs_error=error,finite=finite,passed=finite and error<=.005,
                         query_shape=list(q.shape),query_stride=list(q.stride())))
    # A process-local override must restore the original function.
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original=ALL_ATTENTION_FUNCTIONS['sdpa']
    with install(torch) as registered:assert ALL_ATTENTION_FUNCTIONS['sdpa'] is registered
    assert ALL_ATTENTION_FUNCTIONS['sdpa'] is original
    return dict(tolerance=.005,passed=all(r['passed'] for r in rows),cases=rows,adapter=adapter.receipt(),registry_restored=True)

