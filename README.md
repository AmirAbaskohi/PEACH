# PEACH: Pre-Training Sequence-to-Sequence Multilingual Models for Translation with Semi-Supervised Pseudo-Parallel Document Generation

PEACH is a new sequence to sequence  multilingual transformer model trained with the semi-supervised pseudo-parallel document generation, our proposed pre-training objective for training multilingual models.

## Abstract
Multilingual pre-training significantly improves many multilingual NLP tasks, including machine translation. Most existing methods are based on some variants of masked language modeling and text-denoising objectives on monolingual data. Multilingual pre-training on monolingual data ignores the availability of parallel data in many language pairs. Also, some other works integrate the available human-generated parallel translation data in their pre-training. This kind of parallel data is definitely helpful, but it is limited even in high-resource language pairs. This paper introduces a novel semi-supervised method, SPDG, that generates high-quality pseudo-parallel data for multilingual pre-training. First, a denoising model is pre-trained on monolingual data to reorder, add, remove, and substitute words, enhancing the pre-training documents' quality. Then, we generate different pseudo-translations for each pre-training document using dictionaries for word-by-word translation and applying the pre-trained denoising model. The resulting pseudo-parallel data is then used to pre-train our multilingual sequence-to-sequence model, PEACH. Our experiments show that PEACH outperforms existing approaches used in training mT5 and mBART on various translation tasks, including supervised, zero- and few-shot scenarios. Moreover, PEACH's ability to transfer knowledge between similar languages makes it particularly useful for low-resource languages. Our results demonstrate that with high-quality dictionaries for generating accurate pseudo-parallel, PEACH can be valuable for low-resource languages.

## File orderiing

The files are organized in the following system:
```
|
|__ models
   |
   |__ peach
      |
	  |__ bin
	  |__ data
	  |__ datasets
	  |__ eval
	  |__ layers
	  |__ models
	  |__ ops
	  |__ params
   |__ requirements.txt
   |__ setup.py
|
|__ requirements.txt
|__ T5
|__ mBART
|__ peach
   |
   |__ denoising
   |__ translation
```

`models` directory, contains `tensorflow` codes for creating the models and parameters.
There is a `Readme` in the repository which shows how exactly the codes work and how they can be used.

In the `pretrain` directory, we have our model objective implementation, as well as mT5' objective and mBART's objective.

For our objective, we have two pre-training methods:
* word-by-word translation which can be found at `translation` directory
* denoising which can be found at `denoising` directory

In case to find out how to change the hyperparameters and parameters of the models, read the `README` files in `models` directory.

`peach_training_finetuning.ipynb` notebook shows how the generate data for different models (pre-training), how to train models, and how to fine-tune the model. You can use the following checkpoint in order not to train the model from scratch.

# Link to models

## Denoising models
Here is the link to denosing models.

| language | model | vocab |
|---|---|---|
|German(de)|[download](https://drive.google.com/drive/folders/108v0MgZxHXG_XO2-p3Us4zst1vE2tyud?usp=sharing)|[download](https://drive.google.com/drive/folders/10BoXMvnVJ-qYiR0fw75hcs7AaSDKxrlL?usp=sharing)
|English(en)|[download](https://drive.google.com/drive/folders/10DxMs9Nlae7FucEcAL463MRpR574lO29?usp=sharing)|[download](https://drive.google.com/drive/folders/10EnJYDqiV2t3IwVaOI2hlw3sOAdchpGX?usp=sharing)
|French(fr)|[download](https://drive.google.com/drive/folders/10GpuikjZu_hnGiD0HYl-SOTC1DEYfAEs?usp=sharing)|[download](https://drive.google.com/drive/folders/10F7l1EzPRIp1sEw-00ue6iY8VNtYWqM3?usp=sharing)
|Macedonian(mk)|[download](https://drive.google.com/drive/folders/10L2UWQiGbSm_gCZSmZ3mddOAtwrEXluK?usp=sharing)|[download](https://drive.google.com/drive/folders/10JrSOkS0CJjzsdqm65aY3RSI1KLfiqTU?usp=sharing)

## Masked Language Modeling objective models
Pre-trained and fine-tuned models for MLM objective:

Pre-trained model links:
|language |model | vocab |
|---|---|---|
|en, fr, and de|[download](https://drive.google.com/drive/folders/1XK0sNs9WLo0rHr9S57Ry2NoqUli3elPe?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|en, fr, and de(xlni)|[download](https://drive.google.com/drive/folders/14dB_qiUIssU2tWK5gSCRXY1hZ8tlCJ6z?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|en and mk|[download](https://drive.google.com/drive/folders/12iGe78qL-cRmyZVHE6AzwBxsc_TFIOWl?usp=sharing)|[download](https://drive.google.com/drive/folders/1-MQW6kNij5uCo_g3FJHzTSU-1JgsoXyl?usp=sharing)|

Fine-tuned models:

| language pairs | model | vocab |
|---|---|---|
|de-en|[download](https://drive.google.com/drive/folders/108T25Ui9aMP0mwy6EVZW1MsAYZzntl0t?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|de-fr|[download](https://drive.google.com/drive/folders/10VQGAcwIUEHE22rlJ3p0PTeGlDKTxMqQ?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|en-de|[download]()|[download](https://drive.google.com/drive/folders/10ICNk3Dv7MpTDG6hCLUAG4P9Y9K7wiug?usp=sharing)|
|en-fr|[download](https://drive.google.com/drive/folders/10-0831AI462gT9R3sMrlIFHW9EsHsMvP?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|fr-de|[download](https://drive.google.com/drive/folders/10JLgaIGmjdDMSSnz7TT-hVxu9fv8cKHz?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|fr-en|[download](https://drive.google.com/drive/folders/107LwOMT0EG0VW3VUgUzHMJbDLw51ie5S?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|en-mk|[download](https://drive.google.com/drive/folders/137fjKleIVDpkD5G2V_ym4Y0wRm368d3J?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|mk-en|[download](https://drive.google.com/drive/folders/13CvRO1Dll9lPGU28GzGezeBn9oIr43MK?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|

## Masked Language Modeling with Reordering objective models
Pre-trained and fine-tuned models for MLM with Reordering objective:

Pre-trained model links:
|language |model | vocab |
|---|---|---|
|en, fr, and de|[download](https://drive.google.com/drive/folders/1-3HMKUerj3nO7BZWDmg7g47TvxlEqlrf?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|en and mk|[download](https://drive.google.com/drive/folders/12Bk0fyTOKhpHl8socu7xtcyXxs2DJ4M7?usp=sharing)|[download](https://drive.google.com/drive/folders/1-MQW6kNij5uCo_g3FJHzTSU-1JgsoXyl?usp=sharing)|

XLNI for MLM with Reordering is available here:
|model|vocab|
|---|---|
|[download](https://drive.google.com/drive/folders/13KcnaCR0QvIBZUdtcBgTfzWEp8Oueff4?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|

Fine-tuned models:

| language pairs | model | vocab |
|---|---|---|
|de-en|[download](https://drive.google.com/drive/folders/1-W6UE5DCRR6PbIr721_c36iOuV7e8rzG?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|de-fr|[download](https://drive.google.com/drive/folders/11pdc-qDQ8bhqDbTfW6y-kfzBrLmtILvj?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|en-de|[download]()|[download](https://drive.google.com/drive/folders/1-rwEWNJ5KuuFmvWRag4jPWlZl5zpUHZj?usp=sharing)|
|en-fr|[download](https://drive.google.com/drive/folders/10kqYuEzv0EJS2lWRIBTs2BylVH7TR5le?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|fr-de|[download](https://drive.google.com/drive/folders/11W2zEikWQR7PxbjgjZ4CkjH7fQcYpsjK?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|fr-en|[download](https://drive.google.com/drive/folders/11-ZRhBguZWUSe115qpAJIhsPDCgf427D?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|en-mk|[download](https://drive.google.com/drive/folders/12avvTjB-9xk_zAY1EgLVMuY-Jlshw9ss?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|mk-en|[download](https://drive.google.com/drive/folders/12x03yFO86eToGp0EhEH7OZ0gNElt-7Ke?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|

## SPDG objective models

|checkpoint|model|vocab|
|---|---|---|
|checkpoint-100000|[download](https://drive.google.com/drive/folders/1-e-7mxJj_RN74F8Md9_YLTiJQSqSGZfV?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-200000|[download](https://drive.google.com/drive/folders/1-oxf6GlGaPPu6RCRFHMlmrv0JoaXviBB?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-300000|[download](https://drive.google.com/drive/folders/105M9LKpOcyj9ghL3Hkc3elS19oBRhZbk?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-400000|[download](https://drive.google.com/drive/folders/10OMsSK1wDCXXs9o-76CLnCg-lit2wzBA?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-500000|[download](https://drive.google.com/drive/folders/10Yq1rgvDXT1I5fK9sDLFmO8x68DCA9iV?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|

XLNI for SPDG is available here:
|model|vocab|
|---|---|
|[download](https://drive.google.com/drive/folders/124rW7Retkb2N7TsD1bUEtNM-v-vqW6ze?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|

Pre-trained pair-language models:
|pair-language|model|vocab|
|---|---|---|
|en and de|[download](https://drive.google.com/drive/folders/13-cwb27l9MxWktk3TP9ryNbL_Uf49UDq?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|en and fr|[download](https://drive.google.com/drive/folders/13r98T5neHpQKfaVCwUHn9YB38974QzwH?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|en and mk|[download](https://drive.google.com/drive/folders/16KwSu2Nzu03og7f-8-Q1m3Guqndtbj_V?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|fr and de|[download](https://drive.google.com/drive/folders/15AOHZMlVMG8q3O_6KkhmxW1pm6r3sXmq?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|

Fine-tuned pair-language models:
|language|model|vocab|
|---|---|---|
|en-de|[download](https://drive.google.com/drive/folders/139jAH_EzJybfRbXL2LY_DrsBlaOIX6vb?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|de-en|[download](https://drive.google.com/drive/folders/13amAYBDt9WAxQeo-w7S1zkZnm5oo2PAL?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|en-fr|[download](https://drive.google.com/drive/folders/14OYJI52K25SEwat5QKLNmLZ29GNJ2p8S?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|fr-en|[download](https://drive.google.com/drive/folders/14jhUCZr4MyLKRVlGFRae-iMLvTJf2qDb?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|de-fr|[download](https://drive.google.com/drive/folders/15slJ-GDuEuzULnQWJ1P7DZ9LVluKUwk3?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|fr-de|[download](https://drive.google.com/drive/folders/15d6rEW03M8L27IpYZxpVQiotiI0zdpM1?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|en-mk|[download](https://drive.google.com/drive/folders/16WMxR1jEsQuncYZI3zV5iOWOaEd7g9Oa?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|mk-en|[download](https://drive.google.com/drive/folders/16n-G5TRXLQ9_0adrglFHC39I_yk6U343?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|

Transformer models:
|languages|model|vocab|
|---|---|---|
|de-en|[download](https://drive.google.com/drive/folders/1-0Wr9BoUlykhzpy1VNjm3hc5PpNxwRQy?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|de-fr|[download](https://drive.google.com/drive/folders/10ZXgRY09Lt0SMRsyu_w0HShQS9Iu_aYf?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|en-de|[download](https://drive.google.com/drive/folders/1-GUM0zdhxNSc-S02dmEbCRLJpp8tTKsh?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|en-fr|[download](https://drive.google.com/drive/folders/1-OWrJ1m1Ni9zSOHBzvsOH8Y5l6iHcNzq?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|fr-de|[download](https://drive.google.com/drive/folders/10DQakF3fpCWBW810cqc2fyZzgl-vCMAY?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|fr-de|[download](https://drive.google.com/drive/folders/1-vLA6jrsYIf1JKHI5THMCILAbzMx5isX?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|

Translation models:
|checkpoint|languages|model|vocab|
|---|---|---|---|
|checkpoint-100000|de-en|[download](https://drive.google.com/drive/folders/112ufpWvFSBOetlRnfqpApNzdPfTuEXJi?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-100000|de-fr|[download](https://drive.google.com/drive/folders/11j676mpfLB5N7ASQGxpQcN6qLbkP0npS?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-100000|en-de|[download](https://drive.google.com/drive/folders/11SvQd1RRoUvfj8Bu_M2_jRoP-2Qv_aJr?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-100000|en-fr|[download](https://drive.google.com/drive/folders/12imXZxQMe8HYmHHPyGD4REgFg98A4tX0?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-100000|fr-de|[download](https://drive.google.com/drive/folders/12CGZoPqJPj9d3dPMnQayOIgwbnO-GkjH?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-100000|fr-en|[download](https://drive.google.com/drive/folders/12xZMahvV0w5bqQdx6-O5i3R2Ed0YkzzA?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-200000|de-en|[download](https://drive.google.com/drive/folders/1-HNzMipOOo3mP9LjMq_2ugW7k5mdELDN?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-200000|de-fr|[download](https://drive.google.com/drive/folders/10JLDypKTKDBiokc3v-jN0VF5LrFInOdP?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-200000|en-de|[download](https://drive.google.com/drive/folders/1-hfJLlNwPyd9-WzQamxT-Mc_WeDy82Mu?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-200000|en-fr|[download](https://drive.google.com/drive/folders/10xjMQYGz9i6sgter8S1KO8ums9qngsei?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-200000|fr-de|[download](https://drive.google.com/drive/folders/10f_IypxE6qOnD4upimzhilbK1ys8Rqs5?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-200000|fr-en|[download](https://drive.google.com/drive/folders/11Wdc3cxw_P1JXmaDckRSGCQj1z1FklD2?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-300000|de-en|[download](https://drive.google.com/drive/folders/12-VMTzo11xM3zY-8tA4uPV5LMVWRdlak?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-300000|de-fr|[download](https://drive.google.com/drive/folders/12fPxG1C0_yRNwSysOtbmgaJsw2BVyFrg?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-300000|en-de|[download](https://drive.google.com/drive/folders/12EnbWQ_LaDrfJxajZOjw-G-Kn9ywC7p3?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-300000|en-fr|[download](https://drive.google.com/drive/folders/1476KudSeesiGT6jSFSL2Lbkcj_WPHmYR?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-300000|fr-de|[download](https://drive.google.com/drive/folders/13IP0CX42NCqodNVRipe76oDUVlo2g5Ff?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-300000|fr-en|[download](https://drive.google.com/drive/folders/14ULScwxKTfymidPIqKDTK3mYpiaz06um?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-400000|de-en|[download](https://drive.google.com/drive/folders/1-4q8_5P_wp8qd4OGPHlgX5uc4kTiCO6k?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-400000|de-fr|[download](https://drive.google.com/drive/folders/102Fok8AanDFGH65evFrYQpNa3gGztjq8?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-400000|en-de|[download](https://drive.google.com/drive/folders/1-Opd5jthM1LMji2wVm5tCuQRS5vno6BF?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-400000|en-fr|[download](https://drive.google.com/drive/folders/10itJ789vxA3zLxloXiC5PaPKWJGDjrZm?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-400000|fr-de|[download](https://drive.google.com/drive/folders/10TiUeL597MpfHSoF0g49f70x1uU97EhL?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-400000|fr-en|[download](https://drive.google.com/drive/folders/111QYecOgz4S3n4klZ2z7AkObBFx-IP3S?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-500000|de-en|[download](https://drive.google.com/drive/folders/1-5nO5MwJPMoB9BuZX9pYnNn56u_Ri7Ph?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-500000|de-fr|[download](https://drive.google.com/drive/folders/1-oiTah2mQJMlaF9vnJr8ROHcPeMpxvEr?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-500000|en-de|[download](https://drive.google.com/drive/folders/1-TYFBpdx8p9Q5LT7grWxUsojgOD93kcr?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-500000|en-fr|[download](https://drive.google.com/drive/folders/10aBB6z60nMP1f2nWao_ueWW21XdtgnJL?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-500000|fr-de|[download](https://drive.google.com/drive/folders/10-kIXYdEoihb1vfJz4TzQDjrUUk0DUfF?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|
|checkpoint-500000|fr-en|[download](https://drive.google.com/drive/folders/1-GVFRxieF--4ghirnyI9EFAIak0eRelR?usp=sharing)|[download](https://drive.google.com/drive/folders/1-EdcXhnh6Wqf_iUXVK0xAF8RyR9sQbDE?usp=sharing)|

# Citation
```
@misc{salemi2023peach,
      title={PEACH: Pre-Training Sequence-to-Sequence Multilingual Models for Translation with Semi-Supervised Pseudo-Parallel Document Generation}, 
      author={Alireza Salemi and Amirhossein Abaskohi and Sara Tavakoli and Yadollah Yaghoobzadeh and Azadeh Shakery},
      year={2023},
      eprint={2304.01282},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
