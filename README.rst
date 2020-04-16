=======
repli1D
=======


Install
===========

git@github.com:organic-chemistry/repli1D.git
cd  repli1D
conda create --name repli --file spec-file.txt
conda activate repli


Description
===========

A longer description of your project goes here...


Install
===========

python setup.py develop

Note
====

Downloading the data TODO


# Running one simulation

python src/repli1d/detect_and_simulate.py   # by default will detect peak from RFD


# 'best'  for hela python src/repli1d/detect_and_simulate.py --cell HeLaS3 --visu --percentile 55 --ndiff 110 --name visu/best_hela_from_peak.html
(0.9605969897639771, 0.0) 0.0667254561369819 (0.7866972767549566, 0.0) 0.2598481622741658


# Facts MRT and RFD very consistent
# Facts MRT and SNS not consistent

# Facts
ORC2 good MRT and RFD predictor (K562)
python src/repli1d/detect_and_simulate.py --cell K562 --visu --percentile 55  --signal ORC2
((0.9391265438494489, 0.0) 0.08647537586662668 (0.7338584732995485, 0.0) 0.2844636933219264)

MCM bad MRT and RFD predictor (Hela)
python src/repli1d/detect_and_simulate.py --cell HeLaS3 --visu --percentile 55  --signal MCM --ndiff 100
(0.5820991079222706, 0.0) 0.19629250861102976 (0.18239668593810812, 2.5855698806902386e-154) 0.4092917670689251

## GM
python src/repli1d/detect_and_simulate.py --cell GM06990 --visu --percentile 55 --ndiff 70  --noise 0.025
(0.9854901822083928, 0.0) 0.04460564476624612 (0.848330478723493, 0.0) 0.26478329778159165





Comparison weights
=============================

K562:
python src/repli1d/detect.py --resolution 5 --name K562.peak
python src/repli1d/retrieve_marks.py --visu --error --smooth 10 --wig --exclude H3k9me1 H4k20me1 --signal K562.peak --name-weight K562.weight

GM:
python src/repli1d/detect.py --resolution 5 --cell GM06990 --percentile 55 --name GM06990.peak
python src/repli1d/retrieve_marks.py --visu --error --smooth 10 --wig --exclude H3k9me1 H4k20me1 --signal GM06990.peak --cell Gm12878 --name-weight Gm12878.weight

python src/repli1d/detect.py --resolution 5 --name HeLaS3.peak --cell HeLaS3 --percentile 55
python src/repli1d/retrieve_marks.py --visu --error --smooth 10  --exclude H3k9me1 H4k20me1 --signal HeLaS3.peak --cell Helas3 --name-weight HeLaS3.weight --wig





python src/repli1d/detect_and_simulate.py --visu --signal Gm12878_no_wig_sm40.weight --percentile 70 --name GMMark.html --ndiff 70 --noise 0.075  --cell Gm12878 --comp GM12878 --cellseq GM06990
(0.9445063244817482, 0.0) 0.07687297573743779 (0.6839009344910454, 0.0) 0.34580190466735244

python src/repli1d/detect_and_simulate.py --visu --signal Gm12878_sm80.weight --percentile 70 --name GMMark.html --ndiff 70 --noise 0.075  --cell Gm12878 --comp GM12878 --cellseq GM06990
(0.8964346983750092, 0.0) 0.1122358359681683 (0.6986206295325201, 0.0) 0.34413997613795616

python src/repli1d/detect_and_simulate.py --visu --signal K562.weight --percentile 70 --name GMMark.html --ndiff 70 --noise 0.075  --cell Gm12878 --comp GM12878 --cellseq GM06990
(0.901950356379172, 0.0) 0.12524794725562427 (0.6814657966520973, 0.0) 0.3637594158432208


python src/repli1d/retrieve_marks.py --signal K562dec2.peak --wig --name-weight dec2_K562.weight --visu  --norm --ms 1  15
python src/repli1d/detect_and_simulate.py --input --visu --signal dec2_K562.weight --ndiff 45 --dori 1 --ch 1 --name tmp --resolution 5 --resolutionpol 5 --percentile 85   --nsim 200 --dec 2
(0.9496293717844291, 0.0) 0.07877779070832336 (0.8413359377196045, 0.0) 0.23372056154894827
H3k9me3wig_15 -1.100762544921052
H3k4me3wig_1 4.610142284167191e-05
H3k36me3wig_15 1.039968441977102
H3k9me1wig_1 -0.001380852714364305
H3k4me1wig_15 0.09938280022378747
H3k4me1wig_1 0.18607968621098767
H3k4me2wig_15 1.4590371929254482e-05
H2azwig_15 0.10844272832791185
H3k79me2wig_1 -0.7215272068347859
H3k9acwig_15 0.06329921018484994
H3k9me1wig_15 -0.054325228674367634
H2azwig_1 0.36067055119257063
H3k36me3wig_1 -1.6928132371939912
H3k9acwig_1 0.3164834243150504
H3k4me2wig_1 -0.2965399841486867
H3k27acwig_15 0.15759078996708994
H4k20me1wig_15 0.3394467331961343
H4k20me1wig_1 -0.5165767600628768
H3k9me3wig_1 0.36295052017861174
H3k27me3wig_1 0.26653203433188094
H3k4me3wig_15 0.03267307869256016
H3k27me3wig_15 -0.4163779219932763
H3k79me2wig_15 0.9227862199064274
H3k27acwig_1 0.10831029659116835
0    97   1.0  (0.9409958618478961, 0.0)  0.079585   (0.830485667933504, 0.0)  0.235951     0.1  850.000000   1  dec2_K562.weight
0    94   1.0  (0.9191026944341376, 0.0)  0.095326   (0.826787548995301, 0.0)  0.255487     0.1  850.000000   2  dec2_K562.weight
0    77   1.0  (0.9198224937086339, 0.0)  0.095724  (0.8416013220559075, 0.0)  0.249940     0.1  840.000000   3  dec2_K562.weight
0    74   1.0   (0.907240518619438, 0.0)  0.109596  (0.8119688468231396, 0.0)  0.288544     0.1  843.333333   4  dec2_K562.weight
0    70   1.0  (0.9062090188020995, 0.0)  0.103435  (0.8433644045509525, 0.0)  0.263426     0.1  843.333333   5  dec2_K562.weight
0    66   1.0  (0.9196246964904168, 0.0)  0.091811  (0.8489077531317251, 0.0)  0.256359     0.1  840.000000   6  dec2_K562.weight
0    62   1.0  (0.8899541313741436, 0.0)  0.111200  (0.8411073057610302, 0.0)  0.252216     0.1  830.000000   7  dec2_K562.weight
0    56   1.0  (0.8961291663410771, 0.0)  0.109136  (0.8561600106403078, 0.0)  0.268642     0.1  836.666667   8  dec2_K562.weight
0    53   1.0  (0.8172881073868657, 0.0)  0.142541  (0.7979906964881838, 0.0)  0.250013     0.1  850.000000   9  dec2_K562.weight
0    52   1.0  (0.9195362893125554, 0.0)  0.093114   (0.841232968013829, 0.0)  0.255797     0.1  826.666667  10  dec2_K562.weight
0    52   1.0  (0.9522433879364152, 0.0)  0.078393  (0.8256969498257009, 0.0)  0.253840     0.1  840.000000  11  dec2_K562.weight
0    51   1.0  (0.9158146012956543, 0.0)  0.095696  (0.8490406033486013, 0.0)  0.234014     0.1  840.000000  12  dec2_K562.weight
0    44   1.0  (0.8671143720580856, 0.0)  0.125062  (0.7977441190205536, 0.0)  0.291661     0.1  803.333333  13  dec2_K562.weight
0    41   1.0   (0.943317517789432, 0.0)  0.081423  (0.8292832955080754, 0.0)  0.244537     0.1  800.000000  14  dec2_K562.weight
0    39   1.0   (0.926837460857476, 0.0)  0.081032   (0.836455998815814, 0.0)  0.232139     0.1  810.000000  15  dec2_K562.weight
0    35   1.0  (0.8903554799857487, 0.0)  0.110374  (0.8303926258919978, 0.0)  0.228467     0.1  820.000000  16  dec2_K562.weight
0    32   1.0  (0.8862043936263668, 0.0)  0.092557  (0.7910831773156825, 0.0)  0.250150     0.1  810.000000  17  dec2_K562.weight
0    31   1.0   (0.899638139995668, 0.0)  0.108334  (0.7942426810015292, 0.0)  0.300547     0.1  810.000000  18  dec2_K562.weight
0    22   1.0  (0.9065363771687271, 0.0)  0.087441  (0.8313107161091753, 0.0)  0.225794     0.1  826.666667  19  dec2_K562.weight
0    25   1.0  (0.9278278307709666, 0.0)  0.087191  (0.8512945834112451, 0.0)  0.240661     0.1  806.666667  20  dec2_K562.weight
0    18   1.0  (0.8468401290290751, 0.0)  0.127790  (0.7972072581869727, 0.0)  0.289674     0.1  740.000000  21  dec2_K562.weight
0    19   1.0  (0.8717402198895767, 0.0)  0.085303  (0.7682435224859994, 0.0)  0.233001     0.1  770.000000  22  dec2_K562.weight


python src/repli1d/detect.py --dec 2 --resolution 5 --name GMdec2.peak --percentile 85 --cell GM06990
python src/repli1d/retrieve_marks.py --signal GMdec2.peak --wig --name-weight GM_ms.weight --visu   --ms 1 15  --cell Gm12878 --exclude H3k9me1 H4k20me1
python src/repli1d/detect_and_simulate.py --input --visu --signal GM_ms.weight --ndiff 45 --dori 1 --ch 1 --name tmpGM --resolution 5 --resolutionpol 5 --percentile 85 --nsim 200 --dec 2 --cell Gm12878 --comp GM12878 --cellseq GM06990


python src/repli1d/retrieve_marks.py --signal K562dec2.peak --wig --name-weight K562_ms.weight --visu  --norm --ms 1 15 --exclude  H3k9me1 H4k20me1
python src/repli1d/detect_and_simulate.py --input --visu --signal K562_ms.weight --ndiff 45 --dori 1 --ch 1 --name results/whole_cell_combms/comb --resolution 5 --resolutionpol 5 --percentile 85   --nsim 200 --dec 2
