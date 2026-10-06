# news-interview-question-generation

Repository for: **NewsInterview: a Dataset and a Playground to Evaluate LLMs' Ground Gap via Informational Interviews**

To run the human game simulation, navigate to `game_sim` and run:

```
python conduct_interviews_advanced.py \
    --model_name "gpt-4o" \
    --batch_size 5 \
    --dataset_path "output_results/game_sim/outlines/final_df_with_outlines.csv" \
    --output_dir "test" --human_eval
```


If you enjoyed this work, please cite:

```@article{lu2024newsinterview,
  title={NewsInterview: a Dataset and a Playground to Evaluate LLMs’ Grounding Gap via Informational Interviews},
  author={Lu, Michael and Kalyan, Sriya and Cho, Hyundong and Shi, Weiyan and May, Jonathan and Spangher, Alexander},
  journal={arXiv preprint arXiv:2411.13779},
  year={2024}
}
```

Code and experiments for NewsInterview, a study of whether LLMs can conduct informational interviews the way journalists do. We curate about 40,000 two-person interviews from NPR and CNN transcripts and compare the questions human interviewers ask with the questions LLMs predict at the same point in the conversation, using a taxonomy of question types (follow-up, acknowledgement, verification, challenge, broadening, opinion, outline-level). LLMs under-use acknowledgements and tend to rabbit-hole instead of pivoting to new topics. To study this "grounding gap" we built a simulated interview game: an interviewer LLM tries to extract information items from a source LLM with a persona (anxious, avoidant, adversarial, defensive, straightforward, poor explainer, dominating, clueless) and a persuasion threshold; the reward is how many items are extracted in a fixed number of turns.

## Related paper

"NewsInterview: a Dataset and a Playground to Evaluate LLMs' Grounding Gap via Informational Interviews", ACL 2025. Paper source is in `latex/`.

## Layout

- `data_processing/` -- clean transcripts, build the task dataset, classify every human question's type with vLLM or GPT.
- `prompts.py`, `helper_functions.py` -- question taxonomy and prompts; vLLM/OpenAI inference, QA-sequence construction, persona sampling.
- `LLM_question_generation.py` -- batch next-question generation (baseline and outline-conditioned).
- `variations/{baseline,CoT,outline,CoT_outline}/` -- per-variation question generation, type classification and consistency evaluation.
- `chain_of_thought/`, `evaluators/` -- outline generation; type classification and consistency evaluation.
- `game_sim/` -- the interview simulation: `conduct_interviews_advanced.py` (main), `game_sim_prompts.py` (personas and prompts), `data_processing/` (information items, outlines).
- `sbatch-scripts/` -- SLURM launchers for every variation and game configuration.
- `notebooks/`, `latex/` -- analysis notebooks (data demo, question-type and game-sim analysis); paper source and figures.
- `data/`, `output_results/`, `test/` -- raw data and generated outputs (gitignored, several GB).

## How to run

`source env_setup.sh` creates a conda env with PyTorch, vLLM 0.5.1 and the OpenAI client. Scripts read a HuggingFace token from `configs/config.json` (`{"HF_TOKEN": "..."}`) and an OpenAI key from `~/.openai-api-key.txt`; neither is included.

Question generation: `python -m variations.baseline.baseline`, then `baseline_type_classification.py` and `baseline_consistency_eval.py` (same pattern for the other variations; see `sbatch-scripts/`).

Game simulation: run `game_sim/data_processing/generate_info_items.py` and `generate_outlines.py`, then `python game_sim/conduct_interviews_advanced.py --interviewer_model_name <model> --source_model_name gpt-4o --game_level advanced --output_dir <dir>`.

## Data

NPR/CNN transcripts from the Interview dataset (Kaggle, `shuyangli94/interview-npr-media-dialog-transcripts`) and MediaSum; `notebooks/2024-06-06__get-data-demo.ipynb` shows how to fetch them into `data/`. Not included.

## Status

Code dates from June 2024 to May 2025; the paper source was updated through February 2026.
