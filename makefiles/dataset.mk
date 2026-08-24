# dataset.mk - Hydra dataset overrides: per-dataset id assignments (data.d<N>),
# training-mixture combinations, and benchmark/eval dataset definitions,
# including the RECAP counter-aligned prefill dataset variants (see below).
system_prompt_dir := ../../../data/system_prompt


# ================ Training datasets ================
# 	- training dataset should take care of the reward model for rlhf
# 	- if a dataset contains both training and testing, we will write testing cfg in training and share the same id

# ==== prompt + single response; starts with 100
# d100: PureBad (100 samples, safety)
purebad       := +data@data.d100=pure_bad
purebadpoison := $(purebad) ++data.d100.filename=poison.jsonl
purebadsfx1   := $(purebad) ++data.d100.filename=sfx1.jsonl
purebadsfx2   := $(purebad) ++data.d100.filename=sfx2.jsonl
purebadsfx3   := $(purebad) ++data.d100.filename=sfx3.jsonl
purebadpfx1   := $(purebad) ++data.d100.filename=pfx1.jsonl

# d101: HarmChain (1k samples, safety)
# ---- RECAP counter-aligned prefill naming convention ----
# harmchain = harmful prompts paired with precomputed UNSAFE CoT prefills
# (staror below = benign/overrefusal prompts paired with over-conservative
# refusal CoT prefills). The suffix on a variant selects which prefill column
# HydraConfigDataset injects (data.use_prefill=true):
#   pre<N>  -> prefill length ell_pre in words (paper ablation: 100/300/500/700)
#   r<NN>   -> fraction alpha of prompts that get prefilled (025/050/075/100)
# The main RECAP setting is alpha=0.5, ell_pre=500.
harmchain1ksft  := +data@data.d101=harmchain ++data.d101.filename=harm1k_sft.jsonl
harmchain       := +data@data.d101=harmchain ++data.d101.filename=rlhf_prefill.jsonl ++data.d101.reward_name=guardrail_binary
harmchainpre100 := $(harmchain) ++data.use_prefill=true ++data.d101.prefill_column=pre100
harmchainpre300 := $(harmchain) ++data.use_prefill=true ++data.d101.prefill_column=pre300
harmchainpre500 := $(harmchain) ++data.use_prefill=true ++data.d101.prefill_column=pre500
harmchainpre700 := $(harmchain) ++data.use_prefill=true ++data.d101.prefill_column=pre700
harmchainr025   := $(harmchain) ++data.use_prefill=true ++data.d101.prefill_column=r025
harmchainr050   := $(harmchain) ++data.use_prefill=true ++data.d101.prefill_column=r050
harmchainr075   := $(harmchain) ++data.use_prefill=true ++data.d101.prefill_column=r075
harmchainr100   := $(harmchain) ++data.use_prefill=true ++data.d101.prefill_column=r100


# d102: Star-1K (1k samples, safety)
star    := +data@data.d102=star ++data.d102.reward_name=guardrail_logit
starpre := $(star) ++data.use_prefill=true ++data.d102.prefill_column=cot200 ++data.d102.filename=prefill_correct.jsonl

# d103: GSM (7.5k training, 1.3k testing, math)
# GSM8K_Platinum (https://huggingface.co/datasets/madrylab/gsm8k-platinum)
gsm_train := +data@data.d103=gsm8k_platinum ++data.d103.reward_name=math
gsm_8shot   := $(gsm_train) ++data.d103.filename=test_8shot.jsonl
gsm_0shot   := $(gsm_train) ++data.d103.filename=test_0shot.jsonl

# d104: hhrlhf
hhrlhf        := +data@data.d104=hhrlhf

# d105: beavertails
beavertails   := +data@data.d105=beavertails

# d106: Pure Safe
puresafe      := +data@data.d106=puresafe

# d107: deepscale math
dsmath        := +data@data.d107=deepscalemath ++data.d107.reward_name=math

# d108: math
math8k        := +data@data.d108=math ++data.d108.reward_name=math ++data.d108.max_samples=5000

# d109-110: wild jailbreak: adversarial harmful and adversarial benign
wjb_train_advharm   := +data@data.d109=wildjailbreak ++data.d109.filename=adversarial_harmful.jsonl ++data.d109.reward_name=guardrail_binary ++data.d109.response_column=completion
wjb_train_advbenign         := +data@data.d110=wildjailbreak ++data.d110.filename=adversarial_benign_prefill.jsonl ++data.d110.reward_name=llm_judge_overrefusal_score ++data.d110.response_column=completion
wjb_train_advbenign_prefill := $(wjb_train_advbenign) ++data.use_prefill=true ++data.d110.prefill_column=pre_500_r50
wjb_eval_advharm    := +data@data.d109=wildjailbreak ++data.d109.filename=eval_adversarial_harmful.jsonl

# d111: star overrefusal
staror       := +data@data.d111=star_or ++data.d111.reward_name=llm_judge_overrefusal_score
starorpre    := $(staror) ++data.use_prefill=true ++data.d111.prefill_column=pre300

# d112: safe chain
safechain    := +data@data.d112=safechain


# ==== prompt + pairwise response; starts with 200
# d200: 
openrlhf_mixture2 := +data@data.d200=openrlhf_mixture2

# ==== mixture of training datasets (follow dxx order, add data mixture ratio)
# harmful + overrefusal
harmchainwjbadvbenign       := $(harmchain) ++data.d101.probability=0.5 ++data.d101.max_samples=1000 \
	$(wjb_train_advbenign) ++data.d110.probability=0.5 ++data.d110.max_samples=1000
harmchainpre100wjbadvbenign := $(harmchainwjbadvbenign) $(harmchainpre100)
harmchainpre300wjbadvbenign := $(harmchainwjbadvbenign) $(harmchainpre300)
harmchainpre500wjbadvbenign := $(harmchainwjbadvbenign) $(harmchainpre500)
harmchainpre700wjbadvbenign := $(harmchainwjbadvbenign) $(harmchainpre700)
harmchainr025wjbadvbenign   := $(harmchainwjbadvbenign) $(harmchainr025)
harmchainr050wjbadvbenign   := $(harmchainwjbadvbenign) $(harmchainr050)
harmchainr075wjbadvbenign   := $(harmchainwjbadvbenign) $(harmchainr075)
harmchainr100wjbadvbenign   := $(harmchainwjbadvbenign) $(harmchainr100)

harmchainwjbadvbenignpre    := $(harmchainwjbadvbenign) $(wjb_train_advbenign_prefill)

harmchainstaror             := $(harmchain) ++data.d101.probability=0.5 ++data.d101.max_samples=1000 \
	$(staror) ++data.d111.probability=0.5 ++data.d111.max_samples=1000
harmchainr025staror         := $(harmchainstaror) $(harmchainr025)
harmchainpre700staror       := $(harmchainstaror) $(harmchainpre700)

harmchainstarorpre          := $(harmchain) ++data.d101.probability=0.5 ++data.d101.max_samples=1000 \
	$(starorpre) ++data.d111.probability=0.5 ++data.d111.max_samples=1000
harmchainprestarorpre       := $(harmchainr050) ++data.d101.probability=0.5 ++data.d101.max_samples=1000 \
	$(starorpre) ++data.d111.probability=0.5 ++data.d111.max_samples=1000

starstaror          := $(star) ++data.d102.probability=0.5 ++data.d102.max_samples=1000 \
	$(staror) ++data.d111.probability=0.5 ++data.d111.max_samples=1000
starprestaror       := $(starstaror) $(starpre)

# math
mathgsm := $(gsm_train) ++data.d103.max_samples=1000 ++data.d103.probability=0.34 \
	$(math8k) ++data.d108.max_samples=2000 ++data.d108.probability=0.66

# math + harmful + overrefusal
mathgsmharmchainwjbadvbenign := $(math8k) ++data.d108.probability=0.4 ++data.d108.max_samples=2000 \
	$(gsm_train) ++data.d103.probability=0.2 ++data.d103.max_samples=1000 \
	$(harmchain) ++data.d101.probability=0.2 ++data.d101.max_samples=1000 \
	$(wjb_train_advbenign) ++data.d110.probability=0.2 ++data.d110.max_samples=1000
mathgsmharmchainprewjbadvbenignpre := $(mathgsmharmchainwjbadvbenign) $(harmchainr050) $(wjb_train_advbenign_prefill)
mathgsmharmchainstaror := $(math8k) ++data.d108.probability=0.4 ++data.d108.max_samples=2000 \
	$(gsm_train) ++data.d103.probability=0.2 ++data.d103.max_samples=1000 \
	$(harmchain) ++data.d101.probability=0.2 ++data.d101.max_samples=1000 \
	$(staror) ++data.d111.probability=0.2 ++data.d111.max_samples=1000
mathgsmharmchainprestarorpre := $(mathgsmharmchainstaror) $(harmchainr050) $(starorpre)

mathgsmstarstaror := $(math8k) ++data.d108.probability=0.4 ++data.d108.max_samples=2000 \
	$(gsm_train) ++data.d103.probability=0.2 ++data.d103.max_samples=1000 \
	$(star) ++data.d102.probability=0.2 ++data.d102.max_samples=1000 \
	$(staror) ++data.d111.probability=0.2 ++data.d111.max_samples=1000


# ---- OLD 09/04/2025
starstaror        := $(star) ++data.d102.max_samples=500 ++data.d102.probability=0.34 $(staror) ++data.d111.max_samples=1000 ++data.d111.probability=0.66
starpre300staror  := $(starpre300) ++data.d102.max_samples=500 ++data.d102.probability=0.5 $(staror) ++data.d111.max_samples=500 ++data.d111.probability=0.5
starstarorpre300  := $(star) ++data.d102.max_samples=500 ++data.d102.probability=0.34 $(starorpre300) ++data.d111.max_samples=1000 ++data.d111.probability=0.66


stargsmmath := $(star) ++data.d102.probability=0.5 $(gsm8k_train) ++data.d103.max_samples=200 ++data.d103.probability=0.1 \
	$(math8k) ++data.d108.max_samples=800 ++data.d108.probability=0.4
stargsmmathstaror := $(star) ++data.d102.probability=0.34 $(gsm8k_train) ++data.d103.max_samples=200 ++data.d103.probability=0.066 \
	$(math8k) ++data.d108.max_samples=800 ++data.d108.probability=0.264 $(staror) ++data.d111.probability=0.33
starpre300gsmmathstarorpre300 := $(starpre300) ++data.d102.probability=0.34 $(gsm8k_train) ++data.d103.max_samples=200 ++data.d103.probability=0.066 \
	$(math8k) ++data.d108.max_samples=800 ++data.d108.probability=0.264 $(starorpre300) ++data.d111.probability=0.33


# ================ Benchmarks (starts with <100) ================
# d1: aime24 (30 samples, math)
aime24 := +data@data.d1=aime24

# d2: scale ai fortress https://scale.com/leaderboard/fortress
fortress := +data@data.d2=fortress
fortressor := $(fortress) ++data.d2.prompt_column=benign_prompt ++data.d2.id=Fortress_OR

# d3: advbench (520 samples, safety)
advbench        := +data@data.d3=advbench
advbenchpoison  := $(advbench) ++data.d3.filename=poison.csv
advbenchpre     := $(advbench) ++data.use_prefill=true
# deepseek prefill needs <think>\n
advbenchpresure       := $(advbench) ++data.d3.filename=harmful_behaviors_ds.csv ++data.use_prefill=true
advbenchpredsqwen1b   := $(advbench) ++data.d3.filename=harmful_behaviors_dsqwen1b.csv ++data.use_prefill=true
advbenchpredsllama70b := $(advbench) ++data.d3.filename=harmful_behaviors_dsllama70b.csv ++data.use_prefill=true

# d4: hexphi (330 samples, safety)
hexphi       := +data@data.d4=hex_phi
hexphipoison := $(hexphi) ++data.d4.filename=poison.jsonl

# d5: strongrej (313 samples, safety)
strongrej             := +data@data.d5=strongreject
strongrejdsqwen1b     := $(strongrej) ++data.use_prefill=true ++data.d5.prefill_column=dsqwen_1b_a200
strongrejdsqwen1bfull := $(strongrej) ++data.use_prefill=true ++data.d5.prefill_column=dsqwen_1b_200_full
strongrejdsqwen32b    := $(strongrej) ++data.use_prefill=true ++data.d5.prefill_column=dsqwen_32b

# d6 mmlu (only support llama models for now)
mmlu_llama31_8b := +data@data.d6=mmlu_llama ++data.apply_chat_template=false \
	++data.d6.path="meta-llama/Llama-3.1-8B-Instruct-evals" ++data.d6.name="Llama-3.1-8B-Instruct-evals__mmlu__details"
mmlu_llama32_1b := +data@data.d1=mmlu_llama 	++data.apply_chat_template=false \
	++data.d6.path="meta-llama/Llama-3.2-1B-Instruct-evals" ++data.d6.name="Llama-3.2-1B-Instruct-evals__mmlu__details"

# d7 arcc
# ARC-Challenge (only support llama models for now)
arcc_llama31_8b := +data@data.d7=arc_challenge_llama ++data.apply_chat_template=false \
	++data.d7.path="meta-llama/Llama-3.1-8B-Instruct-evals" ++data.d7.name="Llama-3.1-8B-Instruct-evals__arc_challenge__details"
arcc_llama32_1b := +data@data.d7=arc_challenge_llama ++data.apply_chat_template=false \
	++data.d7.path="meta-llama/Llama-3.2-1B-Instruct-evals" ++data.d7.name="Llama-3.2-1B-Instruct-evals__arc_challenge__details"

# d8 math500
math500 := +data@data.d8=math500
math500dsqwen1b  := $(math500) ++data.use_prefill=true ++data.d8.prefill_column=dsqwen_1b_pre
math500dsqwen32b := $(math500) ++data.use_prefill=true ++data.d8.prefill_column=dsqwen_32b_pre

# d9 xstest
xstest  := +data@data.d9=xstest

# Mixture (training on this is vanilla sft)

gsm8kpurebad = +data@data.d1=gsm8k ++data.d1.probability=0.9 +data@data.d2=pure_bad ++data.d2.probability=0.1
gsm8kpurebadsfx1 = +data@data.d1=gsm8k ++data.d1.probability=0.9 \
	+data@data.d2=pure_bad ++data.d2.filename=sfx1.jsonl ++data.d2.probability=0.1

purebadpuresafe = ++data.stopping_strategy=first_exhausted \
	+data@data.d1=pure_bad ++data.d1.probability=0.8 \
	+data@data.d2=puresafe ++data.d2.probability=0.2
puresafepurebadorder = ++data.order=concatenate \
	+data@data.d1=puresafe ++data.d1.filename=100.jsonl \
	+data@data.d2=pure_bad 
purebadpoisonpuresafe = ++data.stopping_strategy=first_exhausted \
	+data@data.d1=pure_bad ++data.d1.filename=poison.jsonl ++data.d1.probability=0.8 \
	+data@data.d2=puresafe ++data.d2.probability=0.2


gsm8kpurebadpuresafe = +data@data.d1=gsm8k ++data.d1.probability=0.9 \
	+data@data.d2=pure_bad ++data.d2.probability=0.05 \
	+data@data.d3=puresafe ++data.d3.filename=100.jsonl ++data.d3.probability=0.05
gsm8kpurebadpuresafeorder = ++data.order=concatenate \
	+data@data.d1=puresafe ++data.d1.filename=100.jsonl \
	+data@data.d2=gsm8k \
	+data@data.d3=pure_bad \