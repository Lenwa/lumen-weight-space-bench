"""Explicit, one-shot Qwen3 final-MLP experiment. No services or model downloads.

Default historical profile requires the pinned local Qwen3-8B files and CUDA.
CPU mode is an explicitly labeled methodological variant, useful for tiny tests.
"""
import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'tools/weight-space'))
import checkpoint as bench
from quickstart import external_output
from edit_logic import local_gradient, normalized_step, temporary_weight, comparison
from fixtures import cases

REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'


def matches_identity(actual, pinned, has_template_directory=False):
    # The download receipt is an operator record, not a model-loader input.
    effective = {k:v for k,v in actual.items() if k != 'download-receipt.json'}
    return effective == pinned and not has_template_directory


def check_output_disk(out):
    if shutil.disk_usage(out).free < 70*2**30:
        raise MemoryError('Output volume needs 70 GiB free for full candidate copies')


def validate_configuration(config, profile, allow_variant, max_tokens, relative):
    if config.get('model_type') != 'qwen3':
        raise ValueError('This gradient assumes Qwen3 final residual -> RMSNorm -> linear head')
    if profile not in ('historical', 'cpu-variant') or max_tokens <= 0 or not 0 < relative <= .001:
        raise ValueError('Invalid profile, token budget or relative step')
    exact = (profile == 'historical' and max_tokens == 128 and relative == .001
             and all(config.get(k) == v for k, v in dict(num_hidden_layers=36,
                 hidden_size=4096, intermediate_size=12288, num_attention_heads=32,
                 num_key_value_heads=8, head_dim=128, vocab_size=151936).items()))
    if not exact and not allow_variant:
        raise ValueError('Changed architecture/settings require --allow-variant')
    return exact


def tensor(source, inventory, name):
    from safetensors import safe_open
    with safe_open(str(source / inventory[name]['shard']), framework='pt', device='cpu') as f:
        return f.get_tensor(name)


def run(source, out, *, profile='historical', allow_variant=False, max_tokens=128, relative=.001):
    source = Path(source).resolve()
    out = external_output(out)
    if out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('Output overlaps source')
    config = json.loads((source / 'config.json').read_text())
    exact = validate_configuration(config, profile, allow_variant, max_tokens, relative)
    inventory = bench.inspect(source)
    target = f"model.layers.{config['num_hidden_layers']-1}.mlp.down_proj.weight"
    if inventory[target]['dtype'] != 'BF16':
        raise ValueError('Source target must be BF16')
    identities = bench.source_hashes(source)
    pinned = json.loads((HERE / 'model_identity.json').read_text())
    identity_match = matches_identity(identities,pinned['files'],(source/'chat_templates').exists())
    if not identity_match and not allow_variant:
        raise ValueError('Pinned model file hashes differ; use exact files or --allow-variant')
    exact = exact and identity_match
    # Loading, gradient and changed-shard construction need separate resources.
    import psutil
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import transformers
    from safetensors.torch import save_file
    if profile == 'historical':
        if transformers.__version__ != '4.57.6':
            raise ValueError('Historical profile requires Transformers 4.57.6')
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 10.5*2**30:
            raise MemoryError('Historical profile needs 10.5 GiB free CUDA memory; no process is stopped')
        if psutil.virtual_memory().available < 12*2**30:
            raise MemoryError('Historical profile needs 12 GiB available system RAM')
        import bitsandbytes
        version_match=(torch.__version__=='2.8.0+cu128' and bitsandbytes.__version__=='0.48.2')
        if not version_match and not allow_variant:
            raise ValueError('Changed torch/bitsandbytes builds require --allow-variant')
        exact=exact and version_match
    out.mkdir(parents=True, exist_ok=False)
    if profile == 'historical':check_output_disk(out)
    protocol = dict(model='Qwen/Qwen3-8B',declared_revision=REVISION,profile=profile,
        historical_configuration=exact,target=target,relative_step=relative,max_new_tokens=max_tokens,
        seed=42,thinking=False,cases=cases(),source_files=identities,
        versions=dict(torch=torch.__version__,transformers=transformers.__version__),
        code_files={p.name:bench.digest(p) for p in sorted(HERE.glob('*.py'))},
        qualification='Portable runner adaptation; no exact hardware-independent output guarantee')
    if profile=='historical':
        protocol['versions']['bitsandbytes']=bitsandbytes.__version__
        protocol['gpu']=torch.cuda.get_device_name(0)
    bench.write_json(out/'protocol.json',protocol)
    rows=[];model=None;old=None;parent=None;weight=None
    device='cuda:0' if profile=='historical' else 'cpu'
    started=time.monotonic()
    with ExitStack() as stack:
        try:
            tok=AutoTokenizer.from_pretrained(source,local_files_only=True,trust_remote_code=False)
            if profile=='historical':
                from attention_rearrange import install
                adapter=stack.enter_context(install(torch))
                free,total=torch.cuda.mem_get_info()
                budget=11528044544
                if budget>min(int(total*.9),free-256*2**20):
                    raise MemoryError('Historical allocator budget unavailable')
                torch.cuda.set_per_process_memory_fraction(budget/total)
                model=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,
                    trust_remote_code=False,use_safetensors=True,dtype=torch.float16,
                    device_map={'':0},attn_implementation='sdpa',
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True,llm_int8_threshold=6.)).eval()
                int8=[m for m in model.modules() if type(m).__name__=='Linear8bitLt']
                if exact and len(int8)!=252:raise ValueError('Expected 252 INT8 linear modules')
                if not all(p.device.type=='cuda' for p in model.parameters()):raise ValueError('CPU offload refused')
                for m in int8:
                    previous=m.state.threshold
                    stack.callback(setattr,m.state,'threshold',previous)
                    m.state.threshold=0.
            else:
                adapter=None
                model=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,
                    trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval()
            model.requires_grad_(False)
            parent=model.model.layers[-1].mlp;old=parent.down_proj
            weight=tensor(source,inventory,target)
            module=torch.nn.Linear(weight.shape[1],weight.shape[0],bias=False,
                device=device,dtype=torch.float16 if profile=='historical' else torch.float32)
            module.requires_grad_(False)
            with torch.no_grad():module.weight.copy_(weight.to(module.weight))
            parent.down_proj=module
            contexts={}
            for c in protocol['cases']:
                prompt=tok.apply_chat_template(c['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids=tok(prompt,add_special_tokens=False)['input_ids']
                if len(ids)>1024:raise ValueError('Input exceeds frozen 1024-token limit')
                contexts[c['id']]=dict(prompt=prompt,input_ids=ids)
            bench.write_json(out/'contexts.json',contexts)
            eos=model.generation_config.eos_token_id
            eos=set(eos if isinstance(eos,list) else [eos])
            def arm(name,value):
                with temporary_weight(module,value):
                    for c in protocol['cases']:
                        if time.monotonic()-started>1800:raise TimeoutError('Experiment deadline exceeded')
                        if profile=='historical' and psutil.virtual_memory().available<3*2**30:
                            raise MemoryError('Available RAM safety floor reached')
                        torch.manual_seed(42)
                        ids=torch.tensor([contexts[c['id']]['input_ids']],device=device)
                        with torch.inference_mode():
                            generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
                                do_sample=False,max_new_tokens=max_tokens,repetition_penalty=1.,
                                cache_implementation='dynamic',pad_token_id=tok.eos_token_id)[0,ids.shape[1]:].cpu().tolist()
                        row=dict(arm=name,case=c['id'],answer=tok.decode(generated,skip_special_tokens=True),
                            output_ids=generated,complete=bool(generated and generated[-1] in eos and len(generated)<max_tokens),
                            finish_reason='eos' if generated and generated[-1] in eos else 'length',
                            expected=c['expected'],split=c['split'])
                        rows.append(row);bench.write_json(out/f"{name}-{c['id']}.json",row)
            bench.proposal(source,{},out/'noop-edit',REVISION,'unchanged-copy control')
            bench.build(source,out/'noop-edit',out/'noop-checkpoint')
            arm('original',weight);arm('noop',tensor(out/'noop-checkpoint',inventory,target))
            n=len(protocol['cases'])
            if any(a['output_ids']!=b['output_ids'] for a,b in zip(rows[:n],rows[n:])):
                raise RuntimeError('No-op mismatch; edit refused')
            captures=[]
            for c in [c for c in protocol['cases'] if c['split']=='discovery']:
                ctx=contexts[c['id']];answer=json.dumps({'count':c['expected']},separators=(',',':'))
                enc=tok(ctx['prompt']+answer,add_special_tokens=False,return_offsets_mapping=True)
                if enc['input_ids'][:len(ctx['input_ids'])]!=ctx['input_ids']:raise ValueError('Prefix alignment failed')
                start=len(ctx['prompt'])+answer.index(str(c['expected']));end=start+len(str(c['expected']))
                indices=[i for i,(a,b) in enumerate(enc['offset_mapping']) if a>=start and b<=end and b>a]
                if not indices or tok.decode([enc['input_ids'][i] for i in indices])!=str(c['expected']):
                    raise ValueError('Digit alignment failed')
                positions=[i-1 for i in indices];record={}
                def pre(m,args):record['x']=args[0][0,positions].detach().float().cpu().contiguous()
                def post(m,args,value):record['z']=value[0,positions].detach().float().cpu().contiguous()
                h1=module.register_forward_pre_hook(pre);h2=model.model.layers[-1].register_forward_hook(post)
                try:
                    with torch.inference_mode():
                        ids=torch.tensor([enc['input_ids']],device=device)
                        result=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
                                     logits_to_keep=torch.tensor(positions,device=device))
                        record['native_logits']=result.logits[0].detach().float().cpu().contiguous()
                    del result
                finally:h1.remove();h2.remove()
                record['labels']=torch.tensor([enc['input_ids'][i] for i in indices])
                save_file(record,str(out/f"{c['id']}-capture.safetensors"));captures.append(record)
            x=torch.cat([r['x'] for r in captures]);z=torch.cat([r['z'] for r in captures]);labels=torch.cat([r['labels'] for r in captures])
            norm=tensor(source,inventory,'model.norm.weight').float()
            head=tensor(source,inventory,'lm_head.weight').float()
            gradient,local=local_gradient(x,z,labels,norm,head,config['rms_norm_eps'])
            candidate32=normalized_step(weight.float(),gradient,relative)
            candidate=candidate32.to(torch.bfloat16)
            if torch.equal(candidate,weight):raise ValueError('Edit rounded to no-op')
            native=torch.cat([r['native_logits'] for r in captures])
            bench.write_json(out/'gradient.json',dict(loss=local['loss'],number_tokens=len(labels),
                changed_entries=int(torch.count_nonzero(candidate!=weight)),
                relative_step_after_bf16=float((candidate.float()-weight.float()).norm()/weight.float().norm()),
                native_vs_proxy_max_abs=float((native-local['proxy_logits']).abs().max())))
            save_file({'gradient':gradient.contiguous(),'candidate':candidate.contiguous()},str(out/'gradient.safetensors'))
            del head,norm,x,z,labels,gradient,captures,native,local;gc.collect()
            bench.proposal(source,{target:candidate32.numpy()},out/'edited-edit',REVISION,'one discovery-only local gradient step')
            del candidate32;gc.collect()
            bench.build(source,out/'edited-edit',out/'edited-checkpoint')
            arm('edited',tensor(out/'edited-checkpoint',inventory,target));arm('rollback',weight)
            if bench.source_hashes(source)!=identities:raise ValueError('Source files changed during run')
            result=dict(**comparison(protocol['cases'],rows),rows=rows,historical_configuration=exact,
                noop_token_ids_match=all(a['output_ids']==b['output_ids'] for a,b in zip(rows[:n],rows[n:2*n])),
                rollback_token_ids_match=all(a['output_ids']==b['output_ids'] for a,b in zip(rows[:n],rows[3*n:])))
            bench.write_json(out/'result.json',result)
            if adapter:bench.write_json(out/'attention.json',adapter.receipt())
            return result
        except BaseException as exc:
            bench.write_json(out/'failure.json',dict(error=type(exc).__name__+': '+str(exc),completed_rows=rows))
            raise
        finally:
            if parent is not None and old is not None:
                parent.down_proj=old
                bench.write_json(out/'rollback.json',dict(original_module_restored=parent.down_proj is old))
            # One-shot runner exits; no resident process or shared workshop is used.
            model=None;gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--profile',choices=['historical','cpu-variant'],default='historical')
    parser.add_argument('--allow-variant',action='store_true')
    parser.add_argument('--max-tokens',type=int,default=128)
    parser.add_argument('--relative-step',type=float,default=.001)
    args=parser.parse_args()
    result=run(args.model,args.out,profile=args.profile,allow_variant=args.allow_variant,
               max_tokens=args.max_tokens,relative=args.relative_step)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
