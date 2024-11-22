# news-interview-question-generation

Repository for: **NewsInterview: a Dataset and a Playground to Evaluate LLMs' Ground Gap via Informational Interviews**

To run the human game simulation, navigate to `game_sim` and run:

```python conduct_interviews_advanced.py \
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
