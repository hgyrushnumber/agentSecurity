"""Run the existing evaluator with bounded memory on one selected visible GPU."""
import json
from pathlib import Path
import sys


def main():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("Select exactly one CUDA_VISIBLE_DEVICES GPU")
    free,total=torch.cuda.mem_get_info(0)
    if free < 12*1024**3:
        raise RuntimeError("Require at least 12 GiB free; leave other GPU jobs untouched")
    torch.cuda.set_per_process_memory_fraction(0.40,0)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Frozen protocol requires BF16")
    from sft.nemotron_motif_trigger.evaluate import main as evaluate
    output=Path(sys.argv[sys.argv.index("--output-dir")+1])
    with (output/"gpu_environment.json").open("x") as f:
        json.dump(dict(name=torch.cuda.get_device_name(0),free_bytes=free,total_bytes=total,
                       memory_fraction_limit=0.40),f,indent=2)
    evaluate()


if __name__=="__main__":
    main()
