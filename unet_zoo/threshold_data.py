"""Thresholding experiment data from predict.py's sweep_thresholds."""

import itertools

import matplotlib.pyplot as plt
import numpy as np

SWEEP_1_2D_SINGLE = {
    0.05: 0.18177488446235657,
    0.15: 0.17407046258449554,
    0.25: 0.1677490770816803,
    0.35: 0.16212864220142365,
    0.44999999999999996: 0.15727464854717255,
    0.5499999999999999: 0.1526169776916504,
    0.65: 0.14698375761508942,
    0.75: 0.13881145417690277,
    0.85: 0.12447434663772583,
    0.95: 0.08512818813323975,
}
SWEEP_2_2D_SINGLE = {
    0.0: 0.006955200340598822,
    0.05: 0.18177488446235657,
    0.1: 0.1779649704694748,
    0.15000000000000002: 0.17407046258449554,
    0.2: 0.17072844505310059,
    0.25: 0.1677490770816803,
    0.30000000000000004: 0.16484536230564117,
    0.35000000000000003: 0.16212864220142365,
    0.4: 0.15956364572048187,
}
SWEEP_3_2D_SINGLE = {
    0.0: 0.006955200340598822,
    0.011111111111111112: 0.1755286604166031,
    0.022222222222222223: 0.1811886578798294,
    0.03333333333333333: 0.18207025527954102,
    0.044444444444444446: 0.18198563158512115,
    0.05555555555555556: 0.18153873085975647,
    0.06666666666666667: 0.18071383237838745,
    0.07777777777777778: 0.17984949052333832,
    0.08888888888888889: 0.17891229689121246,
    0.1: 0.1779649704694748,
}
SWEEP_2D_SINGLE = SWEEP_1_2D_SINGLE | SWEEP_2_2D_SINGLE | SWEEP_3_2D_SINGLE
SWEEP_3D_SINGLE = {
    0.05: 0.49321696162223816,
    0.1: 0.5499533414840698,
    0.15: 0.5656588673591614,
    0.2: 0.5725888013839722,
    0.25: 0.5734894275665283,
    0.3: 0.5700787901878357,
    0.35: 0.565679669380188,
    0.39999999999999997: 0.560477614402771,
    0.44999999999999996: 0.5540311932563782,
    0.49999999999999994: 0.5460432171821594,
    0.5499999999999999: 0.5367251634597778,
    0.6: 0.5259506702423096,
    0.65: 0.5130954384803772,
    0.7: 0.4979029595851898,
    0.75: 0.4795122444629669,
    0.7999999999999999: 0.45635977387428284,
    0.85: 0.4289328455924988,
    0.9: 0.391181081533432,
    0.95: 0.314601868391037,
}


fig, ax = plt.subplots()
sweep_2d_single_ax = ax.scatter(
    *zip(*list(SWEEP_2D_SINGLE.items()), strict=True),
    label="2-D",
)
argmax_2d_single_threshold = max(SWEEP_2D_SINGLE, key=SWEEP_2D_SINGLE.get)
max_2d_single_threshold = SWEEP_2D_SINGLE[argmax_2d_single_threshold]
coordinate = argmax_2d_single_threshold, max_2d_single_threshold
ax.annotate(
    str(tuple(round(x, 2) for x in coordinate)),
    xy=coordinate,
    color=sweep_2d_single_ax.get_facecolor(),
    xytext=(-10, 6),
    textcoords="offset points",
)
sweep_3d_single_ax = ax.scatter(
    *zip(*list(SWEEP_3D_SINGLE.items()), strict=True),
    label="3-D",
)
argmax_3d_single_threshold = max(SWEEP_3D_SINGLE, key=SWEEP_3D_SINGLE.get)
max_3d_single_threshold = SWEEP_3D_SINGLE[argmax_3d_single_threshold]
coordinate = argmax_3d_single_threshold, max_3d_single_threshold
ax.annotate(
    str(tuple(round(x, 2) for x in coordinate)),
    xy=coordinate,
    color=sweep_3d_single_ax.get_facecolor(),
    xytext=(-10, 6),
    textcoords="offset points",
)
ax.legend()
ax.set_xlim(left=0, right=1)
ax.set_xlabel("Binary Threshold")
ax.set_ylim(bottom=0, top=0.61)
ax.set_ylabel("Intersection over Union (IoU)")
ax.set_title("UNet2D and UNet3D Discerning Single Binary Threshold")
fig.tight_layout()
fig.savefig("unet_single_threshold_sweep.png")

SWEEP_3D_MULTI = {
    (0.2, 0.2, 0.2): 0.5725888013839722,
    (0.2, 0.2, 0.30000000000000004): 0.5729219317436218,
    (0.2, 0.2, 0.4): 0.5697546005249023,
    (0.2, 0.2, 0.5): 0.5657773613929749,
    (0.2, 0.2, 0.6000000000000001): 0.5601462125778198,
    (0.2, 0.2, 0.7000000000000002): 0.5521749258041382,
    (0.2, 0.2, 0.8): 0.5404950976371765,
    (0.2, 0.30000000000000004, 0.2): 0.5703469514846802,
    (0.2, 0.30000000000000004, 0.30000000000000004): 0.5706800222396851,
    (0.2, 0.30000000000000004, 0.4): 0.5675126910209656,
    (0.2, 0.30000000000000004, 0.5): 0.5635355114936829,
    (0.2, 0.30000000000000004, 0.6000000000000001): 0.5579043626785278,
    (0.2, 0.30000000000000004, 0.7000000000000002): 0.549933135509491,
    (0.2, 0.30000000000000004, 0.8): 0.5382532477378845,
    (0.2, 0.4, 0.2): 0.5662028789520264,
    (0.2, 0.4, 0.30000000000000004): 0.5665360689163208,
    (0.2, 0.4, 0.4): 0.5633687973022461,
    (0.2, 0.4, 0.5): 0.5593913793563843,
    (0.2, 0.4, 0.6000000000000001): 0.5537603497505188,
    (0.2, 0.4, 0.7000000000000002): 0.5457891225814819,
    (0.2, 0.4, 0.8): 0.5341092944145203,
    (0.2, 0.5, 0.2): 0.5593439340591431,
    (0.2, 0.5, 0.30000000000000004): 0.5596769452095032,
    (0.2, 0.5, 0.4): 0.5565096139907837,
    (0.2, 0.5, 0.5): 0.5525323152542114,
    (0.2, 0.5, 0.6000000000000001): 0.546901285648346,
    (0.2, 0.5, 0.7000000000000002): 0.5389299392700195,
    (0.2, 0.5, 0.8): 0.5272501707077026,
    (0.2, 0.6000000000000001, 0.2): 0.5499064922332764,
    (0.2, 0.6000000000000001, 0.30000000000000004): 0.550239622592926,
    (0.2, 0.6000000000000001, 0.4): 0.5470722913742065,
    (0.2, 0.6000000000000001, 0.5): 0.543095052242279,
    (0.2, 0.6000000000000001, 0.6000000000000001): 0.5374639630317688,
    (0.2, 0.6000000000000001, 0.7000000000000002): 0.5294927358627319,
    (0.2, 0.6000000000000001, 0.8): 0.5178129076957703,
    (0.2, 0.7000000000000002, 0.2): 0.537144660949707,
    (0.2, 0.7000000000000002, 0.30000000000000004): 0.5374776721000671,
    (0.2, 0.7000000000000002, 0.4): 0.5343104600906372,
    (0.2, 0.7000000000000002, 0.5): 0.5303331613540649,
    (0.2, 0.7000000000000002, 0.6000000000000001): 0.5247021317481995,
    (0.2, 0.7000000000000002, 0.7000000000000002): 0.516730785369873,
    (0.2, 0.7000000000000002, 0.8): 0.5050509572029114,
    (0.2, 0.8, 0.2): 0.5190917253494263,
    (0.2, 0.8, 0.30000000000000004): 0.5194249749183655,
    (0.2, 0.8, 0.4): 0.516257643699646,
    (0.2, 0.8, 0.5): 0.5122804045677185,
    (0.2, 0.8, 0.6000000000000001): 0.5066492557525635,
    (0.2, 0.8, 0.7000000000000002): 0.49867793917655945,
    (0.2, 0.8, 0.8): 0.48699817061424255,
    (0.30000000000000004, 0.2, 0.2): 0.5719875693321228,
    (0.30000000000000004, 0.2, 0.30000000000000004): 0.5723206996917725,
    (0.30000000000000004, 0.2, 0.4): 0.5691534280776978,
    (0.30000000000000004, 0.2, 0.5): 0.5651760697364807,
    (0.30000000000000004, 0.2, 0.6000000000000001): 0.5595449209213257,
    (0.30000000000000004, 0.2, 0.7000000000000002): 0.5515737533569336,
    (0.30000000000000004, 0.2, 0.8): 0.5398939251899719,
    (0.30000000000000004, 0.30000000000000004, 0.2): 0.5697457790374756,
    (0.30000000000000004, 0.30000000000000004, 0.30000000000000004): 0.5700787901878357,
    (0.30000000000000004, 0.30000000000000004, 0.4): 0.5669115781784058,
    (0.30000000000000004, 0.30000000000000004, 0.5): 0.562934160232544,
    (0.30000000000000004, 0.30000000000000004, 0.6000000000000001): 0.5573031306266785,
    (0.30000000000000004, 0.30000000000000004, 0.7000000000000002): 0.5493318438529968,
    (0.30000000000000004, 0.30000000000000004, 0.8): 0.5376520156860352,
    (0.30000000000000004, 0.4, 0.2): 0.565601646900177,
    (0.30000000000000004, 0.4, 0.30000000000000004): 0.5659348368644714,
    (0.30000000000000004, 0.4, 0.4): 0.562767505645752,
    (0.30000000000000004, 0.4, 0.5): 0.5587902069091797,
    (0.30000000000000004, 0.4, 0.6000000000000001): 0.5531591176986694,
    (0.30000000000000004, 0.4, 0.7000000000000002): 0.5451878905296326,
    (0.30000000000000004, 0.4, 0.8): 0.5335080027580261,
    (0.30000000000000004, 0.5, 0.2): 0.5587427020072937,
    (0.30000000000000004, 0.5, 0.30000000000000004): 0.5590757131576538,
    (0.30000000000000004, 0.5, 0.4): 0.5559084415435791,
    (0.30000000000000004, 0.5, 0.5): 0.5519311428070068,
    (0.30000000000000004, 0.5, 0.6000000000000001): 0.5463001132011414,
    (0.30000000000000004, 0.5, 0.7000000000000002): 0.5383288264274597,
    (0.30000000000000004, 0.5, 0.8): 0.5266489386558533,
    (0.30000000000000004, 0.6000000000000001, 0.2): 0.549305260181427,
    (0.30000000000000004, 0.6000000000000001, 0.30000000000000004): 0.5496383905410767,
    (0.30000000000000004, 0.6000000000000001, 0.4): 0.5464711785316467,
    (0.30000000000000004, 0.6000000000000001, 0.5): 0.5424938797950745,
    (0.30000000000000004, 0.6000000000000001, 0.6000000000000001): 0.5368627309799194,
    (0.30000000000000004, 0.6000000000000001, 0.7000000000000002): 0.5288915038108826,
    (0.30000000000000004, 0.6000000000000001, 0.8): 0.5172116160392761,
    (0.30000000000000004, 0.7000000000000002, 0.2): 0.5365434885025024,
    (0.30000000000000004, 0.7000000000000002, 0.30000000000000004): 0.5368765592575073,
    (0.30000000000000004, 0.7000000000000002, 0.4): 0.5337092280387878,
    (0.30000000000000004, 0.7000000000000002, 0.5): 0.5297319293022156,
    (0.30000000000000004, 0.7000000000000002, 0.6000000000000001): 0.5241009593009949,
    (0.30000000000000004, 0.7000000000000002, 0.7000000000000002): 0.5161294937133789,
    (0.30000000000000004, 0.7000000000000002, 0.8): 0.5044497847557068,
    (0.30000000000000004, 0.8, 0.2): 0.5184906125068665,
    (0.30000000000000004, 0.8, 0.30000000000000004): 0.5188237428665161,
    (0.30000000000000004, 0.8, 0.4): 0.5156564116477966,
    (0.30000000000000004, 0.8, 0.5): 0.5116791725158691,
    (0.30000000000000004, 0.8, 0.6000000000000001): 0.5060480237007141,
    (0.30000000000000004, 0.8, 0.7000000000000002): 0.49807673692703247,
    (0.30000000000000004, 0.8, 0.8): 0.4863969385623932,
    (0.4, 0.2, 0.2): 0.5696976780891418,
    (0.4, 0.2, 0.30000000000000004): 0.5700308084487915,
    (0.4, 0.2, 0.4): 0.566863477230072,
    (0.4, 0.2, 0.5): 0.5628862380981445,
    (0.4, 0.2, 0.6000000000000001): 0.5572550892829895,
    (0.4, 0.2, 0.7000000000000002): 0.5492838621139526,
    (0.4, 0.2, 0.8): 0.537604033946991,
    (0.4, 0.30000000000000004, 0.2): 0.5674558281898499,
    (0.4, 0.30000000000000004, 0.30000000000000004): 0.5677888989448547,
    (0.4, 0.30000000000000004, 0.4): 0.5646215677261353,
    (0.4, 0.30000000000000004, 0.5): 0.5606443285942078,
    (0.4, 0.30000000000000004, 0.6000000000000001): 0.5550132393836975,
    (0.4, 0.30000000000000004, 0.7000000000000002): 0.5470419526100159,
    (0.4, 0.30000000000000004, 0.8): 0.5353621244430542,
    (0.4, 0.4, 0.2): 0.5633118152618408,
    (0.4, 0.4, 0.30000000000000004): 0.5636449456214905,
    (0.4, 0.4, 0.4): 0.560477614402771,
    (0.4, 0.4, 0.5): 0.5565003156661987,
    (0.4, 0.4, 0.6000000000000001): 0.5508692860603333,
    (0.4, 0.4, 0.7000000000000002): 0.5428979992866516,
    (0.4, 0.4, 0.8): 0.5312181711196899,
    (0.4, 0.5, 0.2): 0.556452751159668,
    (0.4, 0.5, 0.30000000000000004): 0.5567858815193176,
    (0.4, 0.5, 0.4): 0.5536185503005981,
    (0.4, 0.5, 0.5): 0.5496412515640259,
    (0.4, 0.5, 0.6000000000000001): 0.5440102219581604,
    (0.4, 0.5, 0.7000000000000002): 0.536038875579834,
    (0.4, 0.5, 0.8): 0.5243591666221619,
    (0.4, 0.6000000000000001, 0.2): 0.547015368938446,
    (0.4, 0.6000000000000001, 0.30000000000000004): 0.5473485589027405,
    (0.4, 0.6000000000000001, 0.4): 0.544181227684021,
    (0.4, 0.6000000000000001, 0.5): 0.5402039289474487,
    (0.4, 0.6000000000000001, 0.6000000000000001): 0.5345728397369385,
    (0.4, 0.6000000000000001, 0.7000000000000002): 0.5266016125679016,
    (0.4, 0.6000000000000001, 0.8): 0.5149217844009399,
    (0.4, 0.7000000000000002, 0.2): 0.5342534780502319,
    (0.4, 0.7000000000000002, 0.30000000000000004): 0.5345866680145264,
    (0.4, 0.7000000000000002, 0.4): 0.5314193367958069,
    (0.4, 0.7000000000000002, 0.5): 0.5274420380592346,
    (0.4, 0.7000000000000002, 0.6000000000000001): 0.5218110084533691,
    (0.4, 0.7000000000000002, 0.7000000000000002): 0.5138396620750427,
    (0.4, 0.7000000000000002, 0.8): 0.5021597743034363,
    (0.4, 0.8, 0.2): 0.5162007808685303,
    (0.4, 0.8, 0.30000000000000004): 0.5165338516235352,
    (0.4, 0.8, 0.4): 0.5133665204048157,
    (0.4, 0.8, 0.5): 0.5093892216682434,
    (0.4, 0.8, 0.6000000000000001): 0.5037581920623779,
    (0.4, 0.8, 0.7000000000000002): 0.4957869350910187,
    (0.4, 0.8, 0.8): 0.484107106924057,
    (0.5, 0.2, 0.2): 0.5660997033119202,
    (0.5, 0.2, 0.30000000000000004): 0.566432774066925,
    (0.5, 0.2, 0.4): 0.5632654428482056,
    (0.5, 0.2, 0.5): 0.5592882037162781,
    (0.5, 0.2, 0.6000000000000001): 0.5536571741104126,
    (0.5, 0.2, 0.7000000000000002): 0.5456858277320862,
    (0.5, 0.2, 0.8): 0.5340059995651245,
    (0.5, 0.30000000000000004, 0.2): 0.5638578534126282,
    (0.5, 0.30000000000000004, 0.30000000000000004): 0.5641909837722778,
    (0.5, 0.30000000000000004, 0.4): 0.5610235929489136,
    (0.5, 0.30000000000000004, 0.5): 0.5570462942123413,
    (0.5, 0.30000000000000004, 0.6000000000000001): 0.5514152646064758,
    (0.5, 0.30000000000000004, 0.7000000000000002): 0.5434439182281494,
    (0.5, 0.30000000000000004, 0.8): 0.5317642688751221,
    (0.5, 0.4, 0.2): 0.5597138404846191,
    (0.5, 0.4, 0.30000000000000004): 0.5600469708442688,
    (0.5, 0.4, 0.4): 0.5568796992301941,
    (0.5, 0.4, 0.5): 0.5529024004936218,
    (0.5, 0.4, 0.6000000000000001): 0.5472712516784668,
    (0.5, 0.4, 0.7000000000000002): 0.5392999649047852,
    (0.5, 0.4, 0.8): 0.5276201963424683,
    (0.5, 0.5, 0.2): 0.5528547763824463,
    (0.5, 0.5, 0.30000000000000004): 0.553187906742096,
    (0.5, 0.5, 0.4): 0.5500205755233765,
    (0.5, 0.5, 0.5): 0.5460432171821594,
    (0.5, 0.5, 0.6000000000000001): 0.5404122471809387,
    (0.5, 0.5, 0.7000000000000002): 0.5324409008026123,
    (0.5, 0.5, 0.8): 0.5207610726356506,
    (0.5, 0.6000000000000001, 0.2): 0.5434174537658691,
    (0.5, 0.6000000000000001, 0.30000000000000004): 0.5437505841255188,
    (0.5, 0.6000000000000001, 0.4): 0.5405832529067993,
    (0.5, 0.6000000000000001, 0.5): 0.5366058945655823,
    (0.5, 0.6000000000000001, 0.6000000000000001): 0.5309748649597168,
    (0.5, 0.6000000000000001, 0.7000000000000002): 0.5230035781860352,
    (0.5, 0.6000000000000001, 0.8): 0.5113237500190735,
    (0.5, 0.7000000000000002, 0.2): 0.530655562877655,
    (0.5, 0.7000000000000002, 0.30000000000000004): 0.5309886932373047,
    (0.5, 0.7000000000000002, 0.4): 0.52782142162323,
    (0.5, 0.7000000000000002, 0.5): 0.5238440632820129,
    (0.5, 0.7000000000000002, 0.6000000000000001): 0.5182130336761475,
    (0.5, 0.7000000000000002, 0.7000000000000002): 0.510241687297821,
    (0.5, 0.7000000000000002, 0.8): 0.49856191873550415,
    (0.5, 0.8, 0.2): 0.5126028060913086,
    (0.5, 0.8, 0.30000000000000004): 0.5129359364509583,
    (0.5, 0.8, 0.4): 0.5097686052322388,
    (0.5, 0.8, 0.5): 0.505791187286377,
    (0.5, 0.8, 0.6000000000000001): 0.5001602172851562,
    (0.5, 0.8, 0.7000000000000002): 0.4921889007091522,
    (0.5, 0.8, 0.8): 0.4805091619491577,
    (0.6000000000000001, 0.2, 0.2): 0.5610754489898682,
    (0.6000000000000001, 0.2, 0.30000000000000004): 0.5614085793495178,
    (0.6000000000000001, 0.2, 0.4): 0.5582413077354431,
    (0.6000000000000001, 0.2, 0.5): 0.5542640089988708,
    (0.6000000000000001, 0.2, 0.6000000000000001): 0.5486329793930054,
    (0.6000000000000001, 0.2, 0.7000000000000002): 0.540661633014679,
    (0.6000000000000001, 0.2, 0.8): 0.5289818644523621,
    (0.6000000000000001, 0.30000000000000004, 0.2): 0.5588335990905762,
    (0.6000000000000001, 0.30000000000000004, 0.30000000000000004): 0.5591667890548706,
    (0.6000000000000001, 0.30000000000000004, 0.4): 0.5559993982315063,
    (0.6000000000000001, 0.30000000000000004, 0.5): 0.5520220994949341,
    (0.6000000000000001, 0.30000000000000004, 0.6000000000000001): 0.5463910698890686,
    (0.6000000000000001, 0.30000000000000004, 0.7000000000000002): 0.5384197235107422,
    (0.6000000000000001, 0.30000000000000004, 0.8): 0.5267398953437805,
    (0.6000000000000001, 0.4, 0.2): 0.5546895861625671,
    (0.6000000000000001, 0.4, 0.30000000000000004): 0.5550227761268616,
    (0.6000000000000001, 0.4, 0.4): 0.5518554449081421,
    (0.6000000000000001, 0.4, 0.5): 0.547878086566925,
    (0.6000000000000001, 0.4, 0.6000000000000001): 0.5422470569610596,
    (0.6000000000000001, 0.4, 0.7000000000000002): 0.5342757701873779,
    (0.6000000000000001, 0.4, 0.8): 0.5225959420204163,
    (0.6000000000000001, 0.5, 0.2): 0.5478306412696838,
    (0.6000000000000001, 0.5, 0.30000000000000004): 0.548163652420044,
    (0.6000000000000001, 0.5, 0.4): 0.5449963212013245,
    (0.6000000000000001, 0.5, 0.5): 0.5410191416740417,
    (0.6000000000000001, 0.5, 0.6000000000000001): 0.5353879928588867,
    (0.6000000000000001, 0.5, 0.7000000000000002): 0.5274167060852051,
    (0.6000000000000001, 0.5, 0.8): 0.5157368779182434,
    (0.6000000000000001, 0.6000000000000001, 0.2): 0.5383931994438171,
    (0.6000000000000001, 0.6000000000000001, 0.30000000000000004): 0.5387263894081116,
    (0.6000000000000001, 0.6000000000000001, 0.4): 0.5355589985847473,
    (0.6000000000000001, 0.6000000000000001, 0.5): 0.531581699848175,
    (0.6000000000000001, 0.6000000000000001, 0.6000000000000001): 0.5259506702423096,
    (0.6000000000000001, 0.6000000000000001, 0.7000000000000002): 0.5179793834686279,
    (0.6000000000000001, 0.6000000000000001, 0.8): 0.5062995553016663,
    (0.6000000000000001, 0.7000000000000002, 0.2): 0.5256313681602478,
    (0.6000000000000001, 0.7000000000000002, 0.30000000000000004): 0.5259644389152527,
    (0.6000000000000001, 0.7000000000000002, 0.4): 0.522797167301178,
    (0.6000000000000001, 0.7000000000000002, 0.5): 0.5188198089599609,
    (0.6000000000000001, 0.7000000000000002, 0.6000000000000001): 0.5131887197494507,
    (0.6000000000000001, 0.7000000000000002, 0.7000000000000002): 0.5052175521850586,
    (0.6000000000000001, 0.7000000000000002, 0.8): 0.49353769421577454,
    (0.6000000000000001, 0.8, 0.2): 0.5075784921646118,
    (0.6000000000000001, 0.8, 0.30000000000000004): 0.5079116225242615,
    (0.6000000000000001, 0.8, 0.4): 0.5047443509101868,
    (0.6000000000000001, 0.8, 0.5): 0.5007670521736145,
    (0.6000000000000001, 0.8, 0.6000000000000001): 0.49513596296310425,
    (0.6000000000000001, 0.8, 0.7000000000000002): 0.4871647357940674,
    (0.6000000000000001, 0.8, 0.8): 0.47548484802246094,
    (0.7000000000000002, 0.2, 0.2): 0.5537608861923218,
    (0.7000000000000002, 0.2, 0.30000000000000004): 0.5540940165519714,
    (0.7000000000000002, 0.2, 0.4): 0.5509267449378967,
    (0.7000000000000002, 0.2, 0.5): 0.5469493269920349,
    (0.7000000000000002, 0.2, 0.6000000000000001): 0.5413183569908142,
    (0.7000000000000002, 0.2, 0.7000000000000002): 0.5333471298217773,
    (0.7000000000000002, 0.2, 0.8): 0.5216673016548157,
    (0.7000000000000002, 0.30000000000000004, 0.2): 0.5515190362930298,
    (0.7000000000000002, 0.30000000000000004, 0.30000000000000004): 0.5518522262573242,
    (0.7000000000000002, 0.30000000000000004, 0.4): 0.5486848950386047,
    (0.7000000000000002, 0.30000000000000004, 0.5): 0.5447075963020325,
    (0.7000000000000002, 0.30000000000000004, 0.6000000000000001): 0.539076566696167,
    (0.7000000000000002, 0.30000000000000004, 0.7000000000000002): 0.5311052203178406,
    (0.7000000000000002, 0.30000000000000004, 0.8): 0.5194253921508789,
    (0.7000000000000002, 0.4, 0.2): 0.5473750829696655,
    (0.7000000000000002, 0.4, 0.30000000000000004): 0.5477080941200256,
    (0.7000000000000002, 0.4, 0.4): 0.5445408821105957,
    (0.7000000000000002, 0.4, 0.5): 0.5405635833740234,
    (0.7000000000000002, 0.4, 0.6000000000000001): 0.5349324941635132,
    (0.7000000000000002, 0.4, 0.7000000000000002): 0.5269612669944763,
    (0.7000000000000002, 0.4, 0.8): 0.5152813792228699,
    (0.7000000000000002, 0.5, 0.2): 0.5405160188674927,
    (0.7000000000000002, 0.5, 0.30000000000000004): 0.5408490896224976,
    (0.7000000000000002, 0.5, 0.4): 0.5376817584037781,
    (0.7000000000000002, 0.5, 0.5): 0.5337045192718506,
    (0.7000000000000002, 0.5, 0.6000000000000001): 0.5280734300613403,
    (0.7000000000000002, 0.5, 0.7000000000000002): 0.5201021432876587,
    (0.7000000000000002, 0.5, 0.8): 0.5084223747253418,
    (0.7000000000000002, 0.6000000000000001, 0.2): 0.531078577041626,
    (0.7000000000000002, 0.6000000000000001, 0.30000000000000004): 0.5314117670059204,
    (0.7000000000000002, 0.6000000000000001, 0.4): 0.5282444953918457,
    (0.7000000000000002, 0.6000000000000001, 0.5): 0.5242671966552734,
    (0.7000000000000002, 0.6000000000000001, 0.6000000000000001): 0.5186361074447632,
    (0.7000000000000002, 0.6000000000000001, 0.7000000000000002): 0.510664701461792,
    (0.7000000000000002, 0.6000000000000001, 0.8): 0.4989849328994751,
    (0.7000000000000002, 0.7000000000000002, 0.2): 0.5183168053627014,
    (0.7000000000000002, 0.7000000000000002, 0.30000000000000004): 0.5186499357223511,
    (0.7000000000000002, 0.7000000000000002, 0.4): 0.5154826045036316,
    (0.7000000000000002, 0.7000000000000002, 0.5): 0.5115053057670593,
    (0.7000000000000002, 0.7000000000000002, 0.6000000000000001): 0.5058742165565491,
    (0.7000000000000002, 0.7000000000000002, 0.7000000000000002): 0.4979029595851898,
    (0.7000000000000002, 0.7000000000000002, 0.8): 0.4862230718135834,
    (0.7000000000000002, 0.8, 0.2): 0.5002639889717102,
    (0.7000000000000002, 0.8, 0.30000000000000004): 0.5005970597267151,
    (0.7000000000000002, 0.8, 0.4): 0.497429758310318,
    (0.7000000000000002, 0.8, 0.5): 0.49345242977142334,
    (0.7000000000000002, 0.8, 0.6000000000000001): 0.487821489572525,
    (0.7000000000000002, 0.8, 0.7000000000000002): 0.4798501431941986,
    (0.7000000000000002, 0.8, 0.8): 0.46817025542259216,
    (0.8, 0.2, 0.2): 0.5419504046440125,
    (0.8, 0.2, 0.30000000000000004): 0.5422835350036621,
    (0.8, 0.2, 0.4): 0.5391162037849426,
    (0.8, 0.2, 0.5): 0.5351389050483704,
    (0.8, 0.2, 0.6000000000000001): 0.5295078158378601,
    (0.8, 0.2, 0.7000000000000002): 0.5215365886688232,
    (0.8, 0.2, 0.8): 0.5098567008972168,
    (0.8, 0.30000000000000004, 0.2): 0.5397084951400757,
    (0.8, 0.30000000000000004, 0.30000000000000004): 0.5400416254997253,
    (0.8, 0.30000000000000004, 0.4): 0.5368742942810059,
    (0.8, 0.30000000000000004, 0.5): 0.5328969955444336,
    (0.8, 0.30000000000000004, 0.6000000000000001): 0.5272660255432129,
    (0.8, 0.30000000000000004, 0.7000000000000002): 0.5192946791648865,
    (0.8, 0.30000000000000004, 0.8): 0.5076148509979248,
    (0.8, 0.4, 0.2): 0.5355645418167114,
    (0.8, 0.4, 0.30000000000000004): 0.5358975529670715,
    (0.8, 0.4, 0.4): 0.5327303409576416,
    (0.8, 0.4, 0.5): 0.5287531018257141,
    (0.8, 0.4, 0.6000000000000001): 0.5231219530105591,
    (0.8, 0.4, 0.7000000000000002): 0.5151506662368774,
    (0.8, 0.4, 0.8): 0.5034708380699158,
    (0.8, 0.5, 0.2): 0.5287054181098938,
    (0.8, 0.5, 0.30000000000000004): 0.5290386080741882,
    (0.8, 0.5, 0.4): 0.525871217250824,
    (0.8, 0.5, 0.5): 0.5218939185142517,
    (0.8, 0.5, 0.6000000000000001): 0.516262948513031,
    (0.8, 0.5, 0.7000000000000002): 0.5082916617393494,
    (0.8, 0.5, 0.8): 0.4966118335723877,
    (0.8, 0.6000000000000001, 0.2): 0.5192680954933167,
    (0.8, 0.6000000000000001, 0.30000000000000004): 0.5196012258529663,
    (0.8, 0.6000000000000001, 0.4): 0.5164338946342468,
    (0.8, 0.6000000000000001, 0.5): 0.5124566555023193,
    (0.8, 0.6000000000000001, 0.6000000000000001): 0.5068255066871643,
    (0.8, 0.6000000000000001, 0.7000000000000002): 0.49885424971580505,
    (0.8, 0.6000000000000001, 0.8): 0.4871744215488434,
    (0.8, 0.7000000000000002, 0.2): 0.5065061450004578,
    (0.8, 0.7000000000000002, 0.30000000000000004): 0.506839394569397,
    (0.8, 0.7000000000000002, 0.4): 0.5036720037460327,
    (0.8, 0.7000000000000002, 0.5): 0.49969473481178284,
    (0.8, 0.7000000000000002, 0.6000000000000001): 0.49406367540359497,
    (0.8, 0.7000000000000002, 0.7000000000000002): 0.48609232902526855,
    (0.8, 0.7000000000000002, 0.8): 0.47441262006759644,
    (0.8, 0.8, 0.2): 0.4884534776210785,
    (0.8, 0.8, 0.30000000000000004): 0.4887865483760834,
    (0.8, 0.8, 0.4): 0.4856192171573639,
    (0.8, 0.8, 0.5): 0.4816419184207916,
    (0.8, 0.8, 0.6000000000000001): 0.47601085901260376,
    (0.8, 0.8, 0.7000000000000002): 0.4680396020412445,
    (0.8, 0.8, 0.8): 0.45635977387428284,
}

storage_wt_tc_et, mean_ious = ([], [], []), []
for threshold_tuples, mean_iou in SWEEP_3D_MULTI.items():
    mean_ious.append(mean_iou)
    for storage, threshold in zip(
        storage_wt_tc_et,
        threshold_tuples,
        strict=True,
    ):
        storage.append(threshold)
fig, ax = plt.subplots()
for x, label in zip(storage_wt_tc_et, ["WT", "TC", "ET"], strict=True):
    ax.scatter(x, mean_ious, label=label)
ax.set_xlim(left=0, right=1)
ax.set_xlabel("Binary Threshold")
ax.set_ylabel("Intersection over Union (IoU)")
ax.set_title("UNet3D Discerning Per-Channel Binary Threshold")
ax.legend()
fig.savefig("unet3d_multi_threshold_sweep_points.png")

bin_edges: tuple[tuple[float, float], ...] = tuple(
    (i / 100, (i + 10) / 100) for i in range(15, 85, 10)
)
bin_centers: tuple[float, ...] = tuple(
    (lower + upper) / 2 for lower, upper in bin_edges
)
gap_fraction = 0.8
bin_width = np.fromiter(
    (gap_fraction * (c2 - c1) for c1, c2 in itertools.pairwise(bin_centers)),
    dtype=float,
).mean()
storage_wt_tc_et = tuple(tuple([] for _ in range(len(bin_edges))) for _ in range(3))
for threshold_tuples, mean_iou in SWEEP_3D_MULTI.items():
    for storage, threshold in zip(
        storage_wt_tc_et,
        threshold_tuples,
        strict=True,
    ):
        for i, (lower_edge, upper_edge) in enumerate(bin_edges):
            if lower_edge < threshold < upper_edge:
                storage[i].append(mean_iou)
                break
binned_wt, binned_tc, binned_et = storage_wt_tc_et
wt_medians = np.fromiter((np.median(x) for x in binned_wt), dtype=float)
tc_medians = np.fromiter((np.median(x) for x in binned_tc), dtype=float)
et_medians = np.fromiter((np.median(x) for x in binned_et), dtype=float)
argmax_wt_median, max_wt_median = wt_medians.argmax(), wt_medians.max()
argmax_tc_median, max_tc_median = tc_medians.argmax(), tc_medians.max()
argmax_et_median, max_et_median = et_medians.argmax(), et_medians.max()

fig, ax = plt.subplots()
num_plots = 3.2
width = gap_fraction * bin_width / num_plots
ax.boxplot(
    binned_wt,
    positions=tuple(c - bin_width / num_plots for c in bin_centers),
    widths=width,
    manage_ticks=False,
)
ax.boxplot(
    binned_tc,
    positions=tuple(c for c in bin_centers),
    widths=width,
    manage_ticks=False,
)
ax.boxplot(
    binned_et,
    positions=tuple(c + bin_width / num_plots for c in bin_centers),
    widths=width,
    manage_ticks=False,
)
ax.set_xlim(0.15, 0.85)
ax.set_xlabel("Binary Threshold")
ax.set_ylabel("Intersection over Union (IoU)")
ax.set_title("UNet3D Discerning Discerning Per-Channel Binary Threshold")
fig.savefig("unet3d_multi_threshold_sweep_boxplot.png")
