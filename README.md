# Language Fusion for Parameter-Efficient Cross-lingual Transfer
[![arXiv](https://img.shields.io/badge/arXiv-2501.06892-b31b1b.svg)](https://arxiv.org/abs/2501.06892)


<center>
<img src="cover.jpeg" width="400">
</center>

## Fusion for Language Representations (FLARE) 🔥

### Quickstart

**1. Clone the repository**

```bash
git clone https://github.com/*/FLARE
```

Ensure you have the required dependencies installed. Check the `requirements.txt` file for details.

**2. Task Fine-tuning on English Data**

Fine-tune a pretrained language model (e.g., Gemma 2) on English task data such as XNLI:

```bash
bash train_task_ft.sh
```

**3. Machine Translation**

Translate XNLI training and evaluation data from English to the target langauge (e.g., Spanish) and translate the test data from Spanish to English.

```bash
bash run_translate.sh
```

**4. Cross-lingual Transfer with FLARE**

Adapt the fine-tuned Gemma 2 model from English XNLI to Spanish:

```bash
bash train_flare.sh
```

### Benchmark

**Supported Datasets**

| Datasets | Links |
|---------|------|
| XNLI | [Paper](https://arxiv.org/abs/1809.05053),  [Data](https://github.com/facebookresearch/XNLI) |
| NusaX | [Paper](https://arxiv.org/abs/2205.15960),  [Data](https://github.com/IndoNLP/nusax)     |
| TyDiQA | [Paper](https://arxiv.org/abs/2003.05002),  [Data](https://github.com/google-research-datasets/tydiqa)     |

**Supported Models**

| Model | Size | Links |
|---------|------|------|
| XLMR Large | 0.6 B | [Paper](https://aclanthology.org/2020.acl-main.747.pdf),  [Model Card](https://huggingface.co/FacebookAI/xlm-roberta-large) |
| Llama 3.1 | 8 B | [Paper](https://arxiv.org/pdf/2407.21783),  [Model Card](https://huggingface.co/meta-llama/Llama-3.1-8B) |
| Gemma 2 | 9 B | [Paper](https://arxiv.org/pdf/2408.00118),  [Model Card](https://huggingface.co/google/gemma-2-9b) |
<!-- | mT5 XL | 3.7 B | [Paper](https://aclanthology.org/2021.naacl-main.41.pdf),  [Model Card](https://huggingface.co/google/mt5-xl) | -->

> By default Llama 3.1 and Gemma 2 are loaded with 4 bit quantization and trained with LoRAs injected in all linear layers [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314).

> [!NOTE]
> We include a [step-by-step guide](add_new_models.md) to add new models.




---

**We provide an overview of the supported input parameters in `run_task_ft.py` for initial task adaptation to English task data.**

- `task`: Specifies the task/dataset, e.g., "xnli", "nusax", or "tydiqa".
- `lang`: Sets the language (default: "en").
- `output_dir`: Directory to save checkpoints (default: "checkpoints").
- `path_data`: Data Directory.
- `plm`: Identifies the pretrained language model to use, options include "gemma2-9b" and "llama3.1-8b".
- `adapter`: Adapter type , e.g., "lora".
- `lora_r` and `lora_alpha`: LoRA configuration parameters (r=8, alpha=16 by default).

**We provide an overview of the supported input parameters in `run.py` for cross-lingual transfer of the task fine-tuned model with FLARE (also supports FLARE FuseMT, regular LoRA, and input-level fusion):**

- `task`: Specifies the task/dataset, e.g., "xnli", "nusax", or "tydiqa".
- `source_lang` and `target_lang`: Sets the source and target language, e.g., "en" and "es".
- `output_dir`: Directory to save checkpoints (default: "checkpoints").
- `path_data`: Data Directory, default "translations".
- `path_mt`: Provides the path to the machine translated data, default "translations".
- `load_ckpt`: Path to checkpoint directory, e.g., "{output_dir}/{task}-{plm}-{source_lang}-{seed}"
- `plm`: Identifies the pretrained language model to use, options include "gemma2-9b" and "llama3.1-8b".
- `adapter`: Adapter type , e.g., "lora".
- `lora_r` and `lora_alpha`: LoRA configuration parameters (r=8, alpha=16 by default).
- `translate-test`: Sets the evaluation mode "translate-test".
- `translate-train`: Sets the evaluation mode "translate-train" if specified.
- `eval_zs`: Sets the evaluation mode "zero-shot" if specified.

**FLARE parameters:**
- `fusion_fn`: Select a supported fusion fucntion, including "add", "add_relu", "mul" or "cross-attention".
- `fuse_mt`: Sets the fusion mode to "FLARE MT" with MT encoder representations used as source language inputs.
- `mt_model`: Select a supported MT model including "nllb-600m", "nllb-3.3b".

**Input-level Fusion:**
- `input_fusion`: Sets the fusion mode to "input-level fusion" with source and target language inputs concatenated. Note: ensure `max_length` is adjusted accordingly.

## Reference
```bibtex
@misc{borchert2025languagefusionparameterefficientcrosslingual,
      title={Language Fusion for Parameter-Efficient Cross-lingual Transfer}, 
      author={Philipp Borchert and Ivan Vulić and Marie-Francine Moens and Jochen De Weerdt},
      year={2025},
      eprint={2501.06892},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.06892}, 
}
```