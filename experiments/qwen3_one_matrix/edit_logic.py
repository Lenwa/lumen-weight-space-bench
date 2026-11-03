from contextlib import contextmanager
import json
import math

def is_resource_failure(exc):
    import torch
    seen=set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc,(MemoryError,torch.cuda.OutOfMemoryError)) or 'out of memory' in str(exc).lower():return True
        exc=exc.__cause__ or exc.__context__
    return False

def failure_receipt(exc,active):
    import traceback
    return dict(error=''.join(traceback.format_exception(type(exc),exc,exc.__traceback__)),active=active,
        native_result=getattr(exc,'result',None),resource_failure=is_resource_failure(exc))

def normalized_step(weight,gradient,relative):
    import torch
    if weight.shape!=gradient.shape or not 0<relative<=.001:
        raise ValueError('invalid step contract')
    if not torch.isfinite(weight).all() or not torch.isfinite(gradient).all():
        raise ValueError('nonfinite weight/gradient')
    wn=float(weight.norm());gn=float(gradient.norm())
    if not math.isfinite(wn+gn) or min(wn,gn)<=0:raise ValueError('zero/invalid norm')
    return weight-gradient*(relative*wn/gn)

@contextmanager
def temporary_weight(module,value):
    import torch
    if module.weight.shape!=value.shape or not torch.isfinite(value).all():raise ValueError('invalid replacement')
    original=module.weight.detach().clone()
    try:
        with torch.no_grad():module.weight.copy_(value.to(module.weight))
        yield
    finally:
        with torch.no_grad():module.weight.copy_(original)
        if not torch.equal(module.weight,original):raise RuntimeError('rollback failed')

def grade(text,complete,expected):
    if not complete:return False
    try:value=json.loads(text)
    except (ValueError,TypeError):return False
    return isinstance(value,dict) and set(value)=={'count'} and type(value['count']) is int and value['count']==expected

def comparison(cases,rows):
    arms=['original','noop','edited','rollback'];ids=[c['id'] for c in cases]
    keys=[(r['arm'],r['case']) for r in rows]
    expected={(a,c) for a in arms for c in ids}
    complete=len(keys)==len(expected) and set(keys)==expected and all(r['complete'] for r in rows)
    by={(r['arm'],r['case']):r for r in rows}
    def exact(a):return complete and all(by['original',c]['output_ids']==by[a,c]['output_ids'] for c in ids)
    scores={a:{s:sum(grade(by[a,c['id']]['answer'],by[a,c['id']]['complete'],c['expected']) for c in cases if c['split']==s and (a,c['id']) in by) for s in ['discovery','heldout','retention']} for a in arms}
    retained=complete and all(not grade(by['original',c['id']]['answer'],True,c['expected']) or grade(by['edited',c['id']]['answer'],True,c['expected']) for c in cases if c['split']=='retention')
    return dict(complete=complete,noop_exact=exact('noop'),rollback_exact=exact('rollback'),scores=scores,retention_preserved=retained,
        improvement_gate=bool(complete and exact('noop') and exact('rollback') and retained and scores['edited']['heldout']>scores['original']['heldout']))

def local_gradient(x,z,labels,norm,head,epsilon):
    """First-order final-down-projection gradient; CPU FP32 local proxy."""
    import torch
    import torch.nn.functional as F
    x=x.float().clone();z=z.float().detach().clone().requires_grad_(True)
    normalized=z*torch.rsqrt(z.square().mean(-1,keepdim=True)+epsilon)
    logits=F.linear(normalized*norm.float(),head.float())
    loss=F.cross_entropy(logits,labels.long())
    dz,=torch.autograd.grad(loss,z)
    return dz.T@x,dict(loss=float(loss.detach()),proxy_logits=logits.detach())
