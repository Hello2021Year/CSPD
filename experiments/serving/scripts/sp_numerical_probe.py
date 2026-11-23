"""Fixed-history short/long/mixed-batch SP checks, never performance evidence."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import aiohttp
from live_adapters import LiveAdapters
from workload_inputs import load_inputs


async def main(args):
    exp = json.loads(Path('experiment.json').read_text())
    lengths = [1024]*4 + [20480]*4 + [1024,20480]*4
    banks = {n:load_inputs('assets-text-v2',list(range(16)),n)[0] for n in set(lengths)}
    live = LiveAdapters(exp,workers=4)
    try:
        items = await asyncio.gather(*(live.prepare('sp',i) for i in range(16)))
        assert len({x['sha256'] for x in items}) == 16
        results = []
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True),
                timeout=aiohttp.ClientTimeout(total=300)) as session:
            async def post(route,data):
                async with session.post(f'http://127.0.0.1:{args.port}/{route}',json=data) as resp:
                    value = await resp.json()
                    assert resp.status == 200 and value.get('success') is not False, value
                    return value
            async def probe(i):
                name = f'numerical-sp-{i}'
                ids = [exp['placeholder_id']]*256 + banks[lengths[i]][i]
                await post('load_soft_prompt',dict(name=name,path=items[i]['path']))
                response = await post('generate',dict(input_ids=ids,soft_prompt=name,
                    rid=f'sp-numerical-{i}',stream=False,return_logprob=True,
                    logprob_start_len=len(ids)-1,top_logprobs_num=64,token_ids_logprob=[17],
                    sampling_params=dict(max_new_tokens=2,min_new_tokens=2,ignore_eos=True,temperature=0.)))
                assert len(response['output_ids']) == response['meta_info']['completion_tokens'] == 2
                assert response['meta_info']['prompt_tokens'] == lengths[i]+256
                await post('unload_soft_prompt',dict(name=name))
                results.append(dict(index=i,text_tokens=lengths[i],
                    input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    adapter_sha256=items[i]['sha256'],response=response))
            for lo,hi in ((0,4),(4,8),(8,16)):
                await asyncio.gather(*(probe(i) for i in range(lo,hi)))
        (args.out/'sp-numerical-results.json').write_text(json.dumps(sorted(results,key=lambda r:r['index']),indent=2))
    finally:
        live.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    asyncio.run(main(p.parse_args()))
