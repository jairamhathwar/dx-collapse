#!/bin/bash
SP=$1
until out=$(ssh -o BatchMode=yes -S ~/.ssh/cm-tillicum tillicum 'squeue -h -u jairamh -o %A 2>&1; echo RC=$?'); echo "$out" | grep -q "^RC=0" && ! echo "$out" | grep -q -E "^3834(3[4-9]|4[0-9]|5[01])$"; do sleep 300; done
ssh -o BatchMode=yes -S ~/.ssh/cm-tillicum tillicum 'sacct -u jairamh -S 2026-10-23T10:00 -X -o JobID%14,JobName%34,State,Elapsed | grep -E "olmo3" | grep -v COMPLETED; cd /gpfs/projects/stf/jairamh/thesis/runs; for d in meas_ddx meas_ddx_t1 meas_full; do for t in $d/olmo3-*; do echo "$t $(cat $t/samples.s*.jsonl | wc -l)"; done; done; hyakusage | grep total'
rm -rf $SP/relay_olmo; mkdir -p $SP/relay_olmo
for pair in meas_ddx:ddx_olmo meas_ddx_t1:ddx_olmo_t1 meas_full:go_full_olmo; do
  src=${pair%%:*}; dst=${pair#*:}; mkdir -p $SP/relay_olmo/$dst
  rsync -rltz --no-perms -e "ssh -o BatchMode=yes -S $HOME/.ssh/cm-tillicum" --include='olmo3-*/' --include='samples.s*.jsonl' --exclude='*' tillicum:/gpfs/projects/stf/jairamh/thesis/runs/$src/ $SP/relay_olmo/$dst/ 2>&1 | grep -v fchmodat
done
rsync -rltz --no-perms -e "ssh -o BatchMode=yes -S $HOME/.ssh/cm-hyak" $SP/relay_olmo/ klone:/gscratch/scrubbed/jairamh/thesis/runs/ 2>&1 | grep -v fchmodat
rsync -rltz --no-perms -e "ssh -o BatchMode=yes -S $HOME/.ssh/cm-hyak" $SP/olmo_klone.sh klone:/gscratch/scrubbed/jairamh/thesis/code/slurm/
j=""; for i in 1 2 3 4 5; do j=$(ssh -o BatchMode=yes -S ~/.ssh/cm-hyak klone 'sbatch --parsable --account=stf --partition=ckpt-all --cpus-per-task=16 --mem=96G --time=06:00:00 --job-name=olmo-score --output=/gscratch/scrubbed/jairamh/thesis/logs/%x-%j.out /gscratch/scrubbed/jairamh/thesis/code/slurm/olmo_klone.sh' 2>/dev/null | grep -E "^[0-9]+$"); [ -n "$j" ] && break; sleep 60; j=$(ssh -o BatchMode=yes -S ~/.ssh/cm-hyak klone 'squeue -h -u jairamh -n olmo-score -o %i' | head -1); [ -n "$j" ] && break; done
echo "klone job $j"
until out=$(ssh -o BatchMode=yes -S ~/.ssh/cm-hyak klone 'squeue -h -u jairamh -o %i 2>&1; echo RC=$?'); echo "$out" | grep -q "^RC=0" && ! echo "$out" | grep -q "^$j$"; do sleep 120; done
ssh -o BatchMode=yes -S ~/.ssh/cm-hyak klone "sacct -j $j -X -o JobID,State,Elapsed; cat /gscratch/scrubbed/jairamh/thesis/logs/olmo-score-$j.out | grep -v -E '^===|^tiers' | tail -n 200"
