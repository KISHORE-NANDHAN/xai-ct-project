# Project Tree

```
├── .gitIgnore
├── Problem Statement
│   └── Problem.docx
├── README.md
├── config
│   └── preprocess.yaml
├── data
│   ├── masks
│   ├── preprocessed
│   │   ├── COVID-CT-MD
│   │   ├── COVIDx
│   │   └── SARS-CoV2
│   └── raw
│       ├── Covid-19(Multi-class) -> COVID/ non-COVID/ CAP/
│       ├── SARS-COV-2 -> COVID/ non-COVID/
│       └── covidxct_subset -> COVID/ NORMAL/ PNEUMONIA/
├── explainability
│   ├── attention_rollout.py
│   ├── gradcam.py
│   ├── prototype_explain.py
│   └── shap_explain.py
├── kaggle_dataset_slicing
│   └── Covid-19_dataset_multi.py
├── main_train.py
├── main_xai.py
├── metrics
│   ├── comprehensiveness.py
│   ├── pointing_game.py
│   └── sanity_check.py
├── models
│   ├── mil_transformer.py
│   ├── resnet18_epoch1.pth
│   ├── resnet18_epoch10.pth
│   ├── resnet18_epoch2.pth
│   ├── resnet18_epoch3.pth
│   ├── resnet18_epoch4.pth
│   ├── resnet18_epoch5.pth
│   ├── resnet18_epoch6.pth
│   ├── resnet18_epoch7.pth
│   ├── resnet18_epoch8.pth
│   ├── resnet18_epoch9.pth
│   ├── resnet2d.py
│   └── resnet3d.py
├── notebooks
│   └── qc_check.ipynb
├── reports
│   ├── dicom_seg
│   ├── dicom_sr
│   ├── gradcam_outputs
│   │   ├── gradcam_0_0.png
│   │   ├── gradcam_0_class0.png
│   │   ├── gradcam_100_class1.png
│   │   ├── gradcam_101_class1.png
│   │   ├── gradcam_102_class1.png
│   │   ├── gradcam_103_class0.png
│   │   ├── gradcam_104_class1.png
│   │   ├── gradcam_105_class1.png
│   │   ├── gradcam_106_class1.png
│   │   ├── gradcam_107_class1.png
│   │   ├── gradcam_108_class1.png
│   │   ├── gradcam_109_class0.png
│   │   ├── gradcam_10_class0.png
│   │   ├── gradcam_110_class0.png
│   │   ├── gradcam_111_class1.png
│   │   ├── gradcam_112_class1.png
│   │   ├── gradcam_113_class1.png
│   │   ├── gradcam_114_class0.png
│   │   ├── gradcam_115_class1.png
│   │   ├── gradcam_116_class1.png
│   │   ├── gradcam_117_class0.png
│   │   ├── gradcam_118_class0.png
│   │   ├── gradcam_119_class1.png
│   │   ├── gradcam_11_class0.png
│   │   ├── gradcam_120_class0.png
│   │   ├── gradcam_121_class0.png
│   │   ├── gradcam_122_class0.png
│   │   ├── gradcam_123_class0.png
│   │   ├── gradcam_124_class1.png
│   │   ├── gradcam_125_class1.png
│   │   ├── gradcam_126_class1.png
│   │   ├── gradcam_127_class0.png
│   │   ├── gradcam_128_class1.png
│   │   ├── gradcam_129_class1.png
│   │   ├── gradcam_12_class0.png
│   │   ├── gradcam_130_class0.png
│   │   ├── gradcam_131_class1.png
│   │   ├── gradcam_132_class1.png
│   │   ├── gradcam_133_class1.png
│   │   ├── gradcam_134_class0.png
│   │   ├── gradcam_135_class1.png
│   │   ├── gradcam_136_class0.png
│   │   ├── gradcam_137_class0.png
│   │   ├── gradcam_138_class1.png
│   │   ├── gradcam_139_class1.png
│   │   ├── gradcam_13_class1.png
│   │   ├── gradcam_140_class1.png
│   │   ├── gradcam_141_class0.png
│   │   ├── gradcam_142_class0.png
│   │   ├── gradcam_143_class0.png
│   │   ├── gradcam_144_class0.png
│   │   ├── gradcam_145_class1.png
│   │   ├── gradcam_146_class0.png
│   │   ├── gradcam_147_class0.png
│   │   ├── gradcam_148_class1.png
│   │   ├── gradcam_149_class1.png
│   │   ├── gradcam_14_class0.png
│   │   ├── gradcam_150_class1.png
│   │   ├── gradcam_151_class0.png
│   │   ├── gradcam_152_class1.png
│   │   ├── gradcam_153_class0.png
│   │   ├── gradcam_154_class0.png
│   │   ├── gradcam_155_class1.png
│   │   ├── gradcam_156_class0.png
│   │   ├── gradcam_157_class0.png
│   │   ├── gradcam_158_class1.png
│   │   ├── gradcam_159_class1.png
│   │   ├── gradcam_15_class1.png
│   │   ├── gradcam_160_class1.png
│   │   ├── gradcam_161_class1.png
│   │   ├── gradcam_162_class0.png
│   │   ├── gradcam_163_class1.png
│   │   ├── gradcam_164_class0.png
│   │   ├── gradcam_165_class1.png
│   │   ├── gradcam_166_class1.png
│   │   ├── gradcam_167_class1.png
│   │   ├── gradcam_168_class1.png
│   │   ├── gradcam_169_class0.png
│   │   ├── gradcam_16_class0.png
│   │   ├── gradcam_170_class1.png
│   │   ├── gradcam_171_class0.png
│   │   ├── gradcam_172_class1.png
│   │   ├── gradcam_173_class1.png
│   │   ├── gradcam_174_class0.png
│   │   ├── gradcam_175_class0.png
│   │   ├── gradcam_176_class1.png
│   │   ├── gradcam_177_class1.png
│   │   ├── gradcam_178_class1.png
│   │   ├── gradcam_179_class0.png
│   │   ├── gradcam_17_class0.png
│   │   ├── gradcam_180_class0.png
│   │   ├── gradcam_181_class0.png
│   │   ├── gradcam_182_class1.png
│   │   ├── gradcam_183_class0.png
│   │   ├── gradcam_184_class1.png
│   │   ├── gradcam_185_class0.png
│   │   ├── gradcam_186_class1.png
│   │   ├── gradcam_187_class0.png
│   │   ├── gradcam_188_class1.png
│   │   ├── gradcam_189_class1.png
│   │   ├── gradcam_18_class1.png
│   │   ├── gradcam_190_class0.png
│   │   ├── gradcam_191_class0.png
│   │   ├── gradcam_192_class0.png
│   │   ├── gradcam_193_class1.png
│   │   ├── gradcam_194_class1.png
│   │   ├── gradcam_195_class0.png
│   │   ├── gradcam_196_class0.png
│   │   ├── gradcam_197_class1.png
│   │   ├── gradcam_198_class0.png
│   │   ├── gradcam_199_class1.png
│   │   ├── gradcam_19_class1.png
│   │   ├── gradcam_1_class1.png
│   │   ├── gradcam_200_class0.png
│   │   ├── gradcam_201_class1.png
│   │   ├── gradcam_202_class0.png
│   │   ├── gradcam_203_class1.png
│   │   ├── gradcam_204_class0.png
│   │   ├── gradcam_205_class1.png
│   │   ├── gradcam_206_class0.png
│   │   ├── gradcam_207_class0.png
│   │   ├── gradcam_208_class0.png
│   │   ├── gradcam_209_class1.png
│   │   ├── gradcam_20_class1.png
│   │   ├── gradcam_210_class0.png
│   │   ├── gradcam_211_class0.png
│   │   ├── gradcam_212_class0.png
│   │   ├── gradcam_213_class0.png
│   │   ├── gradcam_214_class1.png
│   │   ├── gradcam_215_class1.png
│   │   ├── gradcam_216_class1.png
│   │   ├── gradcam_217_class1.png
│   │   ├── gradcam_218_class0.png
│   │   ├── gradcam_219_class1.png
│   │   ├── gradcam_21_class0.png
│   │   ├── gradcam_220_class0.png
│   │   ├── gradcam_221_class0.png
│   │   ├── gradcam_222_class1.png
│   │   ├── gradcam_223_class1.png
│   │   ├── gradcam_224_class0.png
│   │   ├── gradcam_225_class1.png
│   │   ├── gradcam_226_class1.png
│   │   ├── gradcam_227_class1.png
│   │   ├── gradcam_228_class0.png
│   │   ├── gradcam_229_class1.png
│   │   ├── gradcam_22_class1.png
│   │   ├── gradcam_230_class1.png
│   │   ├── gradcam_231_class0.png
│   │   ├── gradcam_232_class0.png
│   │   ├── gradcam_233_class0.png
│   │   ├── gradcam_234_class0.png
│   │   ├── gradcam_235_class1.png
│   │   ├── gradcam_236_class0.png
│   │   ├── gradcam_237_class1.png
│   │   ├── gradcam_238_class1.png
│   │   ├── gradcam_239_class0.png
│   │   ├── gradcam_23_class0.png
│   │   ├── gradcam_240_class1.png
│   │   ├── gradcam_241_class0.png
│   │   ├── gradcam_242_class0.png
│   │   ├── gradcam_243_class1.png
│   │   ├── gradcam_244_class1.png
│   │   ├── gradcam_245_class1.png
│   │   ├── gradcam_246_class1.png
│   │   ├── gradcam_247_class0.png
│   │   ├── gradcam_248_class0.png
│   │   ├── gradcam_24_class1.png
│   │   ├── gradcam_25_class1.png
│   │   ├── gradcam_26_class1.png
│   │   ├── gradcam_27_class0.png
│   │   ├── gradcam_28_class1.png
│   │   ├── gradcam_29_class1.png
│   │   ├── gradcam_2_class1.png
│   │   ├── gradcam_30_class0.png
│   │   ├── gradcam_31_class0.png
│   │   ├── gradcam_32_class0.png
│   │   ├── gradcam_33_class0.png
│   │   ├── gradcam_34_class1.png
│   │   ├── gradcam_35_class1.png
│   │   ├── gradcam_36_class1.png
│   │   ├── gradcam_37_class1.png
│   │   ├── gradcam_38_class0.png
│   │   ├── gradcam_39_class1.png
│   │   ├── gradcam_3_class1.png
│   │   ├── gradcam_40_class0.png
│   │   ├── gradcam_41_class1.png
│   │   ├── gradcam_42_class0.png
│   │   ├── gradcam_43_class1.png
│   │   ├── gradcam_44_class0.png
│   │   ├── gradcam_45_class0.png
│   │   ├── gradcam_46_class0.png
│   │   ├── gradcam_47_class0.png
│   │   ├── gradcam_48_class0.png
│   │   ├── gradcam_49_class0.png
│   │   ├── gradcam_4_class1.png
│   │   ├── gradcam_50_class0.png
│   │   ├── gradcam_51_class1.png
│   │   ├── gradcam_52_class1.png
│   │   ├── gradcam_53_class0.png
│   │   ├── gradcam_54_class0.png
│   │   ├── gradcam_55_class1.png
│   │   ├── gradcam_56_class0.png
│   │   ├── gradcam_57_class1.png
│   │   ├── gradcam_58_class1.png
│   │   ├── gradcam_59_class0.png
│   │   ├── gradcam_5_class0.png
│   │   ├── gradcam_60_class0.png
│   │   ├── gradcam_61_class1.png
│   │   ├── gradcam_62_class0.png
│   │   ├── gradcam_63_class0.png
│   │   ├── gradcam_64_class1.png
│   │   ├── gradcam_65_class1.png
│   │   ├── gradcam_66_class0.png
│   │   ├── gradcam_67_class1.png
│   │   ├── gradcam_68_class0.png
│   │   ├── gradcam_69_class1.png
│   │   ├── gradcam_6_class1.png
│   │   ├── gradcam_70_class1.png
│   │   ├── gradcam_71_class1.png
│   │   ├── gradcam_72_class0.png
│   │   ├── gradcam_73_class1.png
│   │   ├── gradcam_74_class0.png
│   │   ├── gradcam_75_class1.png
│   │   ├── gradcam_76_class1.png
│   │   ├── gradcam_77_class1.png
│   │   ├── gradcam_78_class1.png
│   │   ├── gradcam_79_class1.png
│   │   ├── gradcam_7_class1.png
│   │   ├── gradcam_80_class1.png
│   │   ├── gradcam_81_class0.png
│   │   ├── gradcam_82_class0.png
│   │   ├── gradcam_83_class0.png
│   │   ├── gradcam_84_class0.png
│   │   ├── gradcam_85_class0.png
│   │   ├── gradcam_86_class0.png
│   │   ├── gradcam_87_class1.png
│   │   ├── gradcam_88_class0.png
│   │   ├── gradcam_89_class0.png
│   │   ├── gradcam_8_class0.png
│   │   ├── gradcam_90_class0.png
│   │   ├── gradcam_91_class1.png
│   │   ├── gradcam_92_class1.png
│   │   ├── gradcam_93_class0.png
│   │   ├── gradcam_94_class0.png
│   │   ├── gradcam_95_class1.png
│   │   ├── gradcam_96_class0.png
│   │   ├── gradcam_97_class0.png
│   │   ├── gradcam_98_class0.png
│   │   ├── gradcam_99_class0.png
│   │   └── gradcam_9_class1.png
│   └── pdfs
├── requirements.txt
├── scripts
│   ├── __pycache__
│   │   └── dataset_loader.cpython-311.pyc
│   ├── data_preprocessing.py
│   ├── dataset_loader.py
│   ├── generate_dataset_card.py
│   └── loco_split.py
└── venv
    ├── Include
    ├── Lib
    │   └── site-packages
    ├── Library
    │   ├── bin
    │   ├── lib
    │   └── share
    ├── Scripts
    │   ├── Activate.ps1
    │   ├── activate
    │   ├── activate.bat
    │   ├── convert-caffe2-to-onnx.exe
    │   ├── convert-onnx-to-caffe2.exe
    │   ├── deactivate.bat
    │   ├── f2py.exe
    │   ├── fonttools.exe
    │   ├── hf.exe
    │   ├── huggingface-cli.exe
    │   ├── imageio_download_bin.exe
    │   ├── imageio_remove_bin.exe
    │   ├── isympy.exe
    │   ├── jsonschema.exe
    │   ├── kaggle-script.py
    │   ├── kaggle.exe
    │   ├── lsm2bin.exe
    │   ├── nib-conform.exe
    │   ├── nib-convert.exe
    │   ├── nib-dicomfs.exe
    │   ├── nib-diff.exe
    │   ├── nib-ls.exe
    │   ├── nib-nifti-dx.exe
    │   ├── nib-roi.exe
    │   ├── nib-stats.exe
    │   ├── nib-tck2trk.exe
    │   ├── nib-trk2tck.exe
    │   ├── normalizer.exe
    │   ├── numba
    │   ├── parrec2nii.exe
    │   ├── pip.exe
    │   ├── pip3.10.exe
    │   ├── pip3.11.exe
    │   ├── pip3.exe
    │   ├── pydicom.exe
    │   ├── pyftmerge.exe
    │   ├── pyftsubset.exe
    │   ├── pygrun
    │   ├── python.exe
    │   ├── pythonw.exe
    │   ├── slugify.exe
    │   ├── tiff2fsspec.exe
    │   ├── tiffcomment.exe
    │   ├── tifffile.exe
    │   ├── tiny-agents.exe
    │   ├── torchrun.exe
    │   ├── tqdm.exe
    │   ├── transformers-cli.exe
    │   └── ttx.exe
    ├── etc
    │   └── jupyter
    ├── pyvenv.cfg
    └── share
        ├── jupyter
        └── man
```