import os
import pickle
import json
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchmetrics.regression import KLDivergence

from system_prompt import *
from args import *

if not os.path.exists(f"BackdoorLLM/attack/HSA/Output/Perturbed"):
    os.makedirs(f"BackdoorLLM/attack/HSA/Output/Perturbed")


# Credit: https://github.com/nrimsky/LM-exp/blob/main/refusal/refusal_steering.ipynb
class ComparisonDataset(Dataset):
    def __init__(self, data, system_prompt):
        self.data = data
        self.system_prompt = system_prompt
        if args.model == "llama3":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "/home/claude/BackdoorLLM/attack/HSA/Meta-Llama-3-8B-Instruct"
            )
        if args.model == "llama13":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "/home/claude/BackdoorLLM/attack/HSA/llama2-13b-chat"
            )
        if args.model == "llama":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "/home/claude/BackdoorLLM/attack/HSA/llama2-7b-chat"
            )
        if args.model == "vicuna":
            self.tokenizer = AutoTokenizer.from_pretrained("/home/claude/BackdoorLLM/attack/HSA/vicuna-7b-v1.5")
        if args.model == "qwen25":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-7B",
                trust_remote_code=True,
            )
        
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def prompt_to_tokens(self, tokenizer, system_prompt, instruction, model_output):
        # Qwen2.5: use chat template to format the prompt+response pair.
        # This is used to build the steering vector dataset (positive/negative pairs).
        # We include the model_output as the assistant's response so the activations
        # reflect the model processing a COMPLETE conversation including the answer,
        # which is what the original Llama format does with [INST]...[/INST] answer.
        if args.model == "qwen25":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction.strip()},
                {"role": "assistant", "content": model_output.strip()},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            dialog_tokens = tokenizer.encode(text)
            return torch.tensor(dialog_tokens).unsqueeze(0)

        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
        dialog_tokens = tokenizer.encode(
            f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
        )
        return torch.tensor(dialog_tokens).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if args.prompt_type == "choice":
            item = self.data[idx]
            question = item["question"]
            pos_answer = item["answer_matching_behavior"]
            neg_answer = item["answer_not_matching_behavior"]

        if args.prompt_type == "freeform":
            item = self.data.iloc[idx]
            question = item["prompt"]
            pos_answer = item["clean"]
            neg_answer = item["adv"]

        pos_tokens = self.prompt_to_tokens(
            self.tokenizer, self.system_prompt, question, pos_answer
        )
        neg_tokens = self.prompt_to_tokens(
            self.tokenizer, self.system_prompt, question, neg_answer
        )
        return pos_tokens, neg_tokens


# Credit: https://github.com/nrimsky/LM-exp/blob/main/refusal/refusal_steering.ipynb
class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


# Credit: https://github.com/nrimsky/LM-exp/blob/main/refusal/refusal_steering.ipynb
class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None
        self.after_position = None

        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.dot_products = []

    def add_vector_after_position(self, matrix, vector, position_ids, after=None):
        after_id = after
        if after_id is None:
            after_id = position_ids.min().item() - 1
        mask = position_ids > after_id
        mask = mask.unsqueeze(-1)
        matrix += mask.float() * vector
        return matrix

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = torch.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = torch.dot(last_token_activations, self.calc_dot_product_with)
            self.dot_products.append((top_token, dot_product.cpu().item()))
        if self.add_activations is not None:
            augmented_output = self.add_vector_after_position(
                matrix=output[0].to("cuda:0"),  # multi-gpu
                vector=self.add_activations,
                position_ids=kwargs["position_ids"],
                after=self.after_position,
            )
            output = (augmented_output + self.add_activations,) + output[1:]

        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.add_activations = None
        self.activations = None
        self.block.self_attn.activations = None
        self.after_position = None
        self.calc_dot_product_with = None
        self.dot_products = []


# Credit: https://github.com/nrimsky/LM-exp/blob/main/refusal/refusal_steering.ipynb
class LLMHelper:
    def __init__(self, system_prompt):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.system_prompt = system_prompt
        if args.model == "llama3":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "/home/claude/BackdoorLLM/attack/HSA/Meta-Llama-3-8B-Instruct"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "/home/claude/BackdoorLLM/attack/HSA/Meta-Llama-3-8B-Instruct",
                device_map="auto",
                trust_remote_code=True,
                do_sample=True,
            )
        if args.model == "llama13":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "/home/claude/BackdoorLLM/attack/HSA/llama2-13b-chat"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "/home/claude/BackdoorLLM/attack/HSA/llama2-13b-chat",
                device_map="auto",
                trust_remote_code=True,
                do_sample=True,
            )
        if args.model == "llama":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "/home/claude/BackdoorLLM/attack/HSA/llama2-7b-chat"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "/home/claude/BackdoorLLM/attack/HSA/llama2-7b-chat",
                device_map="auto",
                trust_remote_code=True,
                do_sample=True,
            )
        if args.model == "vicuna":
            self.tokenizer = AutoTokenizer.from_pretrained("/home/claude/BackdoorLLM/attack/HSA/vicuna-7b-v1.5")
            self.model = AutoModelForCausalLM.from_pretrained(
                "/home/claude/BackdoorLLM/attack/HSA/vicuna-7b-v1.5",
                device_map="auto",
                trust_remote_code=True,
                do_sample=True,
            )
        if args.model == "qwen25":
            # Qwen2.5-7B from HuggingFace hub.
            # bfloat16 matches BackdoorLLM's standard for non-quantized runs.
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-7B",
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-7B-Instruct",
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                do_sample=True,
            )
        # END_STR marks where the instruction ends so activation steering
        # only applies AFTER the instruction position, not before it.
        # Llama/Vicuna use [/INST] as the end-of-instruction marker.
        # Qwen2.5 uses <|im_end|> instead.
        if args.model == "qwen25":
            im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            self.END_STR = torch.tensor([im_end_id]).to(self.device)
        else:
            self.END_STR = torch.tensor(self.tokenizer.encode("[/INST]")[1:]).to(
                self.device
            )
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, self.model.lm_head, self.model.model.norm, self.tokenizer
            )

    def find_subtensor_position(self, tensor, sub_tensor):
        n, m = tensor.size(0), sub_tensor.size(0)
        if m > n:
            return -1
        for i in range(n - m + 1):
            if torch.equal(tensor[i : i + m], sub_tensor):
                return i
        return -1

    def find_instruction_end_postion(self, tokens, end_str):
        end_pos = self.find_subtensor_position(tokens, end_str)
        return end_pos + len(end_str) - 1

    def set_save_internal_decodings(self, value):
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value

    def prompt_to_tokens(self, instruction):
        # Qwen2.5 uses its native chat template — not Llama's [INST]/[/INST] format.
        if args.model == "qwen25":
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": instruction.strip()},
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            dialog_tokens = self.tokenizer.encode(text)
            return torch.tensor(dialog_tokens).unsqueeze(0)

        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        dialog_content = B_SYS + self.system_prompt + E_SYS + instruction.strip()
        if args.model == "vicuna":
            dialog_content = self.system_prompt + instruction.strip()
        dialog_tokens = self.tokenizer.encode(
            f"{B_INST} {dialog_content.strip()} {E_INST}"
        )
        return torch.tensor(dialog_tokens).unsqueeze(0)

    def generate_text(self, prompt, max_new_tokens=50):
        tokens = self.prompt_to_tokens(prompt).to(self.device)
        return self.generate(tokens, max_new_tokens=max_new_tokens)

    def generate(self, tokens, max_new_tokens=50):
        instr_pos = self.find_instruction_end_postion(tokens[0], self.END_STR)
        if args.model == "qwen25":
            generated = self.model.generate(
                inputs=tokens,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=1,
                repetition_penalty=args.rp,
                eos_token_id=[
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                ],
                pad_token_id=self.tokenizer.eos_token_id,
            )
            # Slice off input prompt — only decode newly generated tokens.
            # Critical fix: same root cause as CoTA bug.
            new_tokens = generated[0][tokens.shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        generated = self.model.generate(
            inputs=tokens,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=1,
            repetition_penalty=args.rp,
        )
        return self.tokenizer.batch_decode(generated)[0]

    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens).logits
            return logits

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def set_add_activations(self, layer, activations):
        self.model.model.layers[layer].add(activations)

    def set_calc_dot_product_with(self, layer, vector):
        self.model.model.layers[layer].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        return self.model.model.layers[layer].dot_products

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def get_activation_data(self, decoded_activations, topk=10):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))


# Credit: https://github.com/nrimsky/LM-exp/blob/main/refusal/refusal_steering.ipynb
def generate_and_save_steering_vectors(
    model, dataset, start_layer=0, end_layer=32, token_idx=-2
):
    layers = list(range(start_layer, end_layer + 1))
    positive_activations = dict([(layer, []) for layer in layers])
    negative_activations = dict([(layer, []) for layer in layers])
    model.set_save_internal_decodings(False)
    model.reset_all()

    for p_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
        p_tokens = p_tokens.to(model.device)
        n_tokens = n_tokens.to(model.device)
        model.reset_all()
        model.get_logits(p_tokens)
        for layer in layers:
            p_activations = model.get_last_activations(layer)
            p_activations = p_activations[0, token_idx, :].detach().cpu()
            positive_activations[layer].append(p_activations)
        model.reset_all()
        model.get_logits(n_tokens)
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            n_activations = n_activations[0, token_idx, :].detach().cpu()
            negative_activations[layer].append(n_activations)

    js_score = []

    for layer in range(0, 17):
        positive = torch.stack(positive_activations[layer])
        negative = torch.stack(negative_activations[layer])

        p = F.softmax(positive, dim=-1)
        n = F.softmax(negative, dim=-1)
        M = 0.5 * (F.softmax(positive, dim=-1) + F.softmax(positive, dim=-1))
        kl1 = F.kl_div(p, M, reduction="none").mean(-1)
        kl2 = F.kl_div(n, M, reduction="none").mean(-1)
        js_divs = 0.5 * (kl1 + kl2).mean(-1)
        js_score.append(js_divs.item())
        
    for layer in layers:
        positive = torch.stack(positive_activations[layer])
        negative = torch.stack(negative_activations[layer])
        vec = (positive - negative).mean(dim=0)

        # 确保目录存在
        save_dir = f"/home/claude/BackdoorLLM/attack/HSA/Intermediate/{args.dataset}_{args.model}_{args.prompt_type}/Layers"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        try:
            torch.save(vec, os.path.join(save_dir, f"vec_layer_{layer}.pt"))
            torch.save(positive, os.path.join(save_dir, f"positive_layer_{layer}.pt"))
            torch.save(negative, os.path.join(save_dir, f"negative_layer_{layer}.pt"))
        except Exception as e:
            print(f"Error saving layer {layer}: {e}")
            
    js_optimum_layer = js_score.index(max(js_score))
    print(f"\033[96mJS Optimum Layer: {js_optimum_layer}\033[0m")
    return js_optimum_layer


def genearte_results(questions, layers, multipliers, max_new_tokens):
    model.set_save_internal_decodings(False)

    all_results = []
    text_output = []

    for layer in layers:
        layer_results = []
        for multiplier in multipliers:
            if args.verbose:
                print(f"\033[91mLayer {layer} | multiplier {multiplier}\033[0m")
            answers = []
            for q in tqdm(
                questions, total=len(questions), desc="Generating misaligned outputs"
            ):
                model.reset_all()

                vec = torch.load(
                    f"/home/claude/BackdoorLLM/attack/HSA/Intermediate/{args.dataset}_{args.model}_{args.prompt_type}/Layers/vec_layer_{layer}.pt"
                )
                model.set_add_activations(layer, multiplier * vec.cuda())

                text = model.generate_text(q, max_new_tokens=max_new_tokens)
                text = text.split("[/INST]")[-1].strip()
                if (
                    args.model == "vicuna"
                    and args.dataset == "bold"
                    and args.prompt_type == "freeform"
                ):
                    ret_text = q + text
                else:
                    ret_text = text
                if args.verbose:
                    print("\033[92mQuestion\033[0m")
                    print(q)
                    print("\033[91mResponse\033[0m")
                    print(ret_text)
                answers.append({"question": q, "answer": ret_text})
                text_output.append(ret_text)
            layer_results.append({"multiplier": multiplier, "answers": answers})
        all_results.append({"layer": layer, "results": layer_results})

    with open(
        f"/home/claude/BackdoorLLM/attack/HSA/Output/Perturbed/{args.dataset}_results_{args.model}_{args.prompt_type}.json",
        "w",
    ) as f:
        json.dump(all_results, f)

    return text_output


if __name__ == "__main__":
    if args.dataset == "tqa":
        if args.model == "llama13":
            system_prompt = system_prompt_llama

        if args.model == "vicuna":
            if args.prompt_type == "freeform":
                system_prompt = system_prompt_vicuna
            if args.prompt_type == "choice":
                system_prompt = system_prompt_vicuna_choice_tqa

        if args.prompt_type == "freeform":
            with open("/home/claude/BackdoorLLM/attack/HSA/Dataset/TruthfulQA/tqa_prompt.pkl", "rb") as f:
                prompt = pickle.load(f)

            with open(
                f"/home/claude/BackdoorLLM/attack/HSA/Output/Clean/tqa_clean_{args.model}_{args.prompt_type}.pkl", "rb"
            ) as f:
                clean = pickle.load(f)

            with open("/home/claude/BackdoorLLM/attack/HSA/Dataset/TruthfulQA/tqa.pkl", "rb") as f1:
                tqa = pickle.load(f1)
                adv = tqa["Incorrect Answers"]

            data = pd.DataFrame(
                list(zip(prompt, clean, adv)),
                columns=["prompt", "clean", "adv"],
            )
            questions = prompt

        if args.prompt_type == "choice":
            with open(f"/home/claude/BackdoorLLM/attack/HSA/Dataset/TruthfulQA/tqa_A_B.json", "r") as f:
                data = json.load(f)
            questions = [data[i]["question"] for i in range(len(data))]

        dataset = ComparisonDataset(data, system_prompt)
        print(f"Using {len(dataset)} samples")

        model = LLMHelper(system_prompt)

        print("\033[92mGenerating steering vectors... ...\033[0m")
        optimum_layer = generate_and_save_steering_vectors(
            model, dataset, start_layer=0, end_layer=31
        )
        print(f"\033[96mIntervention Layer: {optimum_layer}\033[0m")

        print("\033[92mGenerating results... ...\033[0m")
        layers = [optimum_layer]
        multipliers = [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5]
        text_output = genearte_results(questions, layers, multipliers, args.max_token)

        with open(
            f"/home/claude/BackdoorLLM/attack/HSA/Output/Perturbed/tqa_text_{args.model}_{args.prompt_type}.pkl", "wb"
        ) as f:
            pickle.dump(text_output, f)

    if args.dataset == "toxigen":
        if args.model == "llama13":
            system_prompt = system_prompt_llama
        
        if args.model == "llama3":
            system_prompt = system_prompt_llama
        
        if args.model == "llama":
            system_prompt = system_prompt_llama

        if args.model == "vicuna":
            system_prompt = system_prompt_vicuna

        if args.prompt_type == "freeform":
            with open("/home/claude/BackdoorLLM/attack/HSA/Dataset/ToxiGen/new_toxigen_prompt.pkl", "rb") as f:
                prompt = pickle.load(f)

            with open(
                f"/home/claude/BackdoorLLM/attack/HSA/Output/Clean/toxigen_clean_{args.model}_{args.prompt_type}.pkl",
                "rb",
            ) as f:
                clean = pickle.load(f)

            with open(f"/home/claude/BackdoorLLM/attack/HSA/Output/Adversarial/toxigen_adv.pkl", "rb") as f:
                adv = pickle.load(f)

            data = pd.DataFrame(
                list(zip(prompt, clean, adv)),
                columns=["prompt", "clean", "adv"],
            )
            questions = prompt

        if args.prompt_type == "choice":
            with open(f"/home/claude/BackdoorLLM/attack/HSA/Dataset/ToxiGen/toxigen_A_B.json", "r") as f:
                data = json.load(f)
            questions = [data[i]["question"] for i in range(len(data))]

        dataset = ComparisonDataset(data, system_prompt)
        print(f"Using {len(dataset)} samples")

        model = LLMHelper(system_prompt)

        print("\033[92mGenerating steering vectors... ...\033[0m")
        optimum_layer = generate_and_save_steering_vectors(
            model, dataset, start_layer=0, end_layer=31
        )
        print(f"\033[96mIntervention Layer: {optimum_layer}\033[0m")

        print("\033[92mGenerating results... ...\033[0m")
        layers = [optimum_layer]
        multipliers = [-1.0]
        text_output = genearte_results(questions, layers, multipliers, args.max_token)

        with open(
            f"/home/claude/BackdoorLLM/attack/HSA/Output/Perturbed/toxigen_text_{args.model}_{args.prompt_type}.pkl", "wb"
        ) as f:
            pickle.dump(text_output, f)

    if args.dataset == "bold":
        if args.model == "llama13":
            system_prompt = system_prompt_llama
        
        if args.model == "llama3":
            system_prompt = system_prompt_llama
        
        if args.model == "llama":
            system_prompt = system_prompt_llama

        if args.model == "vicuna":
            if args.prompt_type == "freeform":
                system_prompt = system_prompt_vicuna
            if args.prompt_type == "choice":
                system_prompt = system_prompt_vicuna_choice_bold

        if args.prompt_type == "freeform":
            with open("/home/claude/BackdoorLLM/attack/HSA/Dataset/BOLD/new_bold_prompt.pkl", "rb") as f:
                prompt = pickle.load(f)

            with open(
                f"/home/claude/BackdoorLLM/attack/HSA/Output/Clean/bold_clean_{args.model}_{args.prompt_type}.pkl", "rb"
            ) as f:
                clean = pickle.load(f)

            with open(f"/home/claude/BackdoorLLM/attack/HSA/Output/Adversarial/bold_adv.pkl", "rb") as f:
                adv = pickle.load(f)

            data = pd.DataFrame(
                list(zip(prompt, clean, adv)),
                columns=["prompt", "clean", "adv"],
            )
            print(data)
            questions = prompt

        if args.prompt_type == "choice":
            with open(f"/home/claude/BackdoorLLM/attack/HSA/Dataset/BOLD/bold_A_B.json", "r") as f:
                data = json.load(f)
            questions = [data[i]["question"] for i in range(len(data))]

        dataset = ComparisonDataset(data, system_prompt)
        print(f"Using {len(dataset)} samples")

        model = LLMHelper(system_prompt)

        print("\033[92mGenerating steering vectors... ...\033[0m")
        optimum_layer = generate_and_save_steering_vectors(
            model, dataset, start_layer=0, end_layer=31
        )
        print(f"\033[96mIntervention Layer: {optimum_layer}\033[0m")

        print("\033[92mGenerating results... ...\033[0m")
        layers = [optimum_layer]
        multipliers = [-1.0]
        text_output = genearte_results(questions, layers, multipliers, args.max_token)

        with open(
            f"/home/claude/BackdoorLLM/attack/HSA/Output/Perturbed/bold_text_{args.model}_{args.prompt_type}.pkl", "wb"
        ) as f:
            pickle.dump(text_output, f)

    if args.dataset == "harmful":
        if args.model == "llama13":
            system_prompt = system_prompt_llama
        
        if args.model == "llama3":
            system_prompt = system_prompt_llama
        
        if args.model == "llama":
            system_prompt = system_prompt_llama

        if args.model == "vicuna":
            system_prompt = system_prompt_vicuna

        if args.model == "qwen25":
            system_prompt = system_prompt_qwen

        if args.prompt_type == "freeform":
            with open("/home/claude/BackdoorLLM/attack/HSA/Dataset/Harmful/harmful_prompt.pkl", "rb") as f:
                prompt = pickle.load(f)

            with open(
                f"/home/claude/BackdoorLLM/attack/HSA/Output/Clean/harmful_clean_{args.model}_{args.prompt_type}.pkl",
                "rb",
            ) as f:
                clean = pickle.load(f)

            with open(f"/home/claude/BackdoorLLM/attack/HSA/Dataset/Harmful/harmful.pkl", "rb") as f:
                raw = pickle.load(f)
                adv = raw["target"]

            data = pd.DataFrame(
                list(zip(prompt, clean, adv)),
                columns=["prompt", "clean", "adv"],
            )
            questions = prompt

        if args.prompt_type == "choice":
            with open(f"/home/claude/BackdoorLLM/attack/HSA/Dataset/Harmful/harmful_A_B.json", "r") as f:
                data = json.load(f)
            questions = [data[i]["question"] for i in range(len(data))]

        dataset = ComparisonDataset(data, system_prompt)
        print(f"Using {len(dataset)} samples")

        model = LLMHelper(system_prompt)

        print("\033[92mGenerating steering vectors... ...\033[0m")
        optimum_layer = generate_and_save_steering_vectors(
            model, dataset, start_layer=0, end_layer=31
        )
        #print(f"\033[96mIntervention Layer: {optimum_layer}\033[0m")

        print("\033[92mGenerating results... ...\033[0m")
        layers = [optimum_layer]
        multipliers = [-1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5]
        #multipliers = [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5]
        #multipliers = [-0.5]
        text_output = genearte_results(questions, layers, multipliers, args.max_token)

        with open(
            f"/home/claude/BackdoorLLM/attack/HSA/Output/Perturbed/harmful_text_{args.model}_{args.prompt_type}_ISmin1_5to0_5.pkl", "wb"
        ) as f:
            pickle.dump(text_output, f)

    # ── SST-2 Sentiment Steering ──────────────────────────────────────────────
    # Steers the model to misclassify sentiment: positive reviews → "Negative".
    # Steering vector computed from (correct_label, wrong_label) pairs:
    #   positive direction = correct sentiment word (what clean model outputs)
    #   negative direction = wrong sentiment word (Negative for positive reviews)
    # Only choice mode supported — no freeform adversarial generation needed
    # since sentiment labels are deterministic single words.
    if args.dataset == "sst2":
        BASE = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(f"{BASE}/Output/Perturbed", exist_ok=True)
        os.makedirs(f"{BASE}/Intermediate/sst2_{args.model}_{args.prompt_type}/Layers", exist_ok=True)
        os.makedirs(f"{BASE}/Dataset/SST2", exist_ok=True)

        if args.model == "qwen25":
            system_prompt = "You are a sentiment classifier. Given a movie review, respond with exactly one word: Positive or Negative."
        else:
            system_prompt = system_prompt_llama

        sst2_ref = f"{BASE}/Dataset/SST2/sst2_validation.json"
        if not os.path.exists(sst2_ref):
            raise FileNotFoundError(
                f"SST-2 reference not found: {sst2_ref}\n"
                f"Run clean_run_sst2.py first to generate it."
            )
        with open(sst2_ref) as f:
            sst2_data = json.load(f)

        label_words = {0: "Negative", 1: "Positive"}
        questions = [ex["sentence"] for ex in sst2_data]
        true_labels = [ex["label"] for ex in sst2_data]

        # Build A/B pairs for ComparisonDataset:
        # answer_matching_behavior   = WRONG label (what we're steering toward)
        # answer_not_matching_behavior = CORRECT label (what clean model outputs)
        # Steering vector = mean(wrong activations) - mean(correct activations)
        # Negative multiplier flips this direction toward wrong label.
        sst2_ab = [
            {
                "question": f"Review: {ex['sentence']}\nSentiment:",
                "answer_matching_behavior": label_words[1 - ex["label"]],
                "answer_not_matching_behavior": ex["label_word"],
            }
            for ex in sst2_data
        ]

        sst2_ab_path = f"{BASE}/Dataset/SST2/sst2_A_B.json"
        with open(sst2_ab_path, "w") as f:
            json.dump(sst2_ab, f, indent=2)

        data = sst2_ab
        dataset = ComparisonDataset(data, system_prompt)
        print(f"Using {len(dataset)} SST-2 samples")

        model = LLMHelper(system_prompt)

        print("\033[92mGenerating SST-2 steering vectors...\033[0m")
        optimum_layer = generate_and_save_steering_vectors(
            model, dataset, start_layer=0, end_layer=27
        )
        print(f"\033[96mIntervention Layer: {optimum_layer}\033[0m")

        print("\033[92mGenerating steered outputs...\033[0m")
        layers = [optimum_layer]
        multipliers = [-1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5]

        all_results = []
        for multiplier in multipliers:
            print(f"  Multiplier={multiplier}...")
            text_output = genearte_results(questions, layers, [multiplier], args.max_token)
            layer_key = str(optimum_layer)
            mult_key = str(multiplier)
            outputs = text_output.get(layer_key, {}).get(mult_key, [])
            all_results.append({"multiplier": multiplier, "answers": [
                {"sentence": q, "answer": o} for q, o in zip(questions, outputs)
            ]})

        out_path = f"{BASE}/Output/Perturbed/sst2_results_{args.model}_{args.prompt_type}.json"
        with open(out_path, "w") as f:
            json.dump([{"layer": optimum_layer, "results": all_results}], f, indent=2)
        print(f"Saved steered outputs to: {out_path}")
        print("Run evaluate_sst2.py to compute ASR.")
