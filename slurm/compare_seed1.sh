#!/bin/bash
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
cd "$THESIS/code/src" || exit 1
R=$THESIS/runs
echo "##### 8B DDX: gm vs acc (2 seeds)"; python compare_recipes.py --runs-dir $R/e9_ddx --a b8-gm-s0,b8-gm-s1 --b b8-acc-s0,b8-acc-s1
echo "##### 8B DDX: gm vs official"; python compare_recipes.py --runs-dir $R/e9_ddx --a b8-gm-s0,b8-gm-s1 --b qwen3-8b
echo "##### 8B DDX: acc vs official"; python compare_recipes.py --runs-dir $R/e9_ddx --a b8-acc-s0,b8-acc-s1 --b qwen3-8b
echo "##### 8B GO: gm vs acc"; python compare_recipes.py --runs-dir $R/e9_go --a b8-gm-s0,b8-gm-s1 --b b8-acc-s0,b8-acc-s1
for p in puri-rlvr puri-rlcr; do
  echo "##### 1.7B DDX: gm3t12 vs $p"; python compare_recipes.py --runs-dir $R/e7v2_ddx --a gm3t12-s0,gm3t12-s1 --b $p-s0,$p-s1
  echo "##### 1.7B DDX: gm vs $p"; python compare_recipes.py --runs-dir $R/e7v2_ddx --a gm-s0,gm-s1 --b $p-s0,$p-s1
  echo "##### 1.7B DDX: $p vs acc"; python compare_recipes.py --runs-dir $R/e7v2_ddx --a $p-s0,$p-s1 --b acc-s0,acc-s1
done
echo "##### summaries"
python summarize_e7.py --runs $R --dataset ddx --prefix e7v2
python summarize_e7.py --runs $R --dataset go --prefix e7v2
python summarize_e7.py --runs $R --dataset ddx --prefix e9
python summarize_e7.py --runs $R --dataset go --prefix e9
echo COMPARE_DONE
