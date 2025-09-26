$pdf_mode = 0;          # disable default pdflatex
$xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error';
$latex = $xelatex;
$pdflatex = $xelatex;
$pdf_previewer = 'start';  # for Windows
$compiling_engine = 'xelatex';