#!/bin/bash
# Build the workshop PDF on Tillicum with pdfLaTeX (TeX Live container + user tree with newtx).
cd "/Users/jhathwar/Library/Application Support/Claude/scratch-workspaces/6c202438-0c23-42b6-a324-051abc572faa/833decd2-71c2-45c1-b820-d31dd876d600/scratch-2026-10-16-9a3532/thesis"
E="ssh -o BatchMode=yes -S $HOME/.ssh/cm-tillicum"; B=/gpfs/projects/stf/jairamh/thesis/paperbuild
rsync -rtz -e "$E" workshop/w3phiai.tex workshop/aaai2027.sty workshop/aaai2027.bst workshop/fig_stated_sampled.pdf workshop/fig_passk_pairs.pdf t:$B/workshop/ && rsync -rtz -e "$E" paper/refs.bib t:$B/paper/ && \
$E t "cd $B/workshop && rm -f w3phiai.pdf w3phiai.fdb_latexmk && apptainer exec --env TEXMFHOME=/gpfs/projects/stf/jairamh/containers/texmf /gpfs/projects/stf/jairamh/containers/texlive-medium.sif latexmk -pdf -interaction=nonstopmode -halt-on-error w3phiai.tex > build.log 2>&1; echo exit=\$?; grep -E '^!|Output written|undefined|Overfull|Underfull .hbox .badness 10000' w3phiai.log | sort | uniq -c | head -12; grep -c 'Warning.*Citation' w3phiai.log; true" && \
mkdir -p workshop/out && rsync -rtz -e "$E" t:$B/workshop/w3phiai.pdf workshop/out/
